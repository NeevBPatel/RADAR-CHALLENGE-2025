import numpy as np
import scipy as scp
import adi
from adi.adar1000 import adar1000_array
from adi.adf4159 import adf4159
from target_detection_dbfs import cfar
import helpers
from scipy.signal.windows import taylor
import cmath
import cupy as cp
import cupyx.scipy.signal
import warnings
import matplotlib.pyplot as plt
import time
from enum import Enum
from typing import List, Dict, Tuple, Optional
import logging


class target():
    
    def __init__(self, number = 0, angle = 0.0):
        """
        A custom class to define a target object with enhanced tracking capabilities.
        """
        self.target_number = number
        self.range_estimate = 0  # meters
        self.angle_estimate = angle  # degrees
        self.track_record = []  # Past record of target movement. Will help for future prediction
        self.future_predict = []  # Future prediction. Will help to track object pre-emptively
        self.received_data = []
        self.signal_freq = 0
        self.freq_index = 0
        
        # Tracking attributes
        self.power_threshold = -25  # dBFS threshold for target loss detection
        self.current_power = 0  # Current received power
        self.detected_power = 0  # updated everytime new data is received 
        self.power_history = []  # History of received powers
        self.frequency_deviation_threshold = 10e3  # 5kHz frequency deviation threshold
        self.angle_search_range = 20  # degrees for beam sweep
        self.target_lost_flag = False  # Flag indicating target loss
        self.frequency_shifted = False  # Flag indicating frequency shift
        self.last_valid_angle = angle  # Last known good angle
        self.last_valid_freq = 0  # Last known good frequency
        self.detected_freq = 0  # updated everytime new data is received
        self.consecutive_loss_count = 0  # Counter for consecutive target losses
        self.max_filter_bandwidth = 30e3  # Maximum filter bandwidth
        
        
    def receive_data(self, my_phaser, my_sdr):
        data = my_sdr.rx()
        self.received_data = data
        return data
        
    def digi_phase_shift(self, data, phase):
        
        '''
            Apply Digital Phase Shift to given Data
        '''
        
        phase_delay = np.exp(1j*np.deg2rad(phase))
        shifted_data = data * phase_delay
        
        return shifted_data
    
    
    def dbfs_cupy(self, raw_data):
        """
        Converts IQ samples to a FFT plot, scaled in dBFS, using Cupy for GPU acceleration.

        Args:
            raw_data (np.ndarray): A NumPy array of raw IQ samples.

        Returns:
            np.ndarray: A NumPy array representing the FFT plot in dBFS.
        """
        raw_data_cp = cp.asarray(raw_data)
        win_funct_cp = cupyx.scipy.signal.windows.taylor(len(raw_data_cp), nbar=10, sll=200)
        y_cp = raw_data_cp * win_funct_cp
        sp_cp = cp.absolute(cp.fft.fft(y_cp))
        #sp_cp = cp.fft.fftshift(sp_cp)
        s_mag_cp = cp.abs(sp_cp) / cp.sum(win_funct_cp)
        s_mag_cp = cp.maximum(s_mag_cp, 10**-15)
        s_dbfs_cp = 20 * cp.log10(s_mag_cp / (2**11))
        s_dbfs_np = cp.asnumpy(s_dbfs_cp)
        return s_dbfs_np

        
    def dbfs(self, raw_data):
        # function to convert IQ samples to FFT plot, scaled in dBFS
        win_funct = taylor(len(raw_data),nbar=10,sll=200)
        y = raw_data * win_funct
        sp = np.absolute(np.fft.fft(y))
        #sp = np.fft.fftshift(sp)
        s_mag = np.abs(sp) / np.sum(win_funct)
        s_mag = np.maximum(s_mag, 10 ** (-15))
        s_dbfs = 20 * np.log10(s_mag / (2 ** 11))
        return  s_dbfs
        
    
    def process_target_data(self, data_dbfs, Indices, freqs, dist, guard_cells, ref_cells, bias, method):
        main_data = data_dbfs[Indices]
        peaks, targets = cfar(main_data, guard_cells, ref_cells, bias, method)
        
        num_groups, groups = helpers.find_groups_of_consecutive_false(targets.mask)

        est_dist = []
        est_freq = []
        est_power = []
        for i in range(0,num_groups):
            indices = groups[i]
            if indices[0] == indices[1]:
                x = np.average(dist[indices[0]], weights=main_data[indices[0]])
                y = np.average(freqs[indices[0]], weights=main_data[indices[0]])
                z = np.average(main_data[indices[0]], weights=main_data[indices[0]])
            else:
                x = np.average(dist[indices[0]:indices[1]], weights=main_data[indices[0]:indices[1]])
                y = np.average(freqs[indices[0]:indices[1]], weights=main_data[indices[0]:indices[1]])
                z = np.average(main_data[indices[0]:indices[1]], weights=main_data[indices[0]:indices[1]])
            est_dist.append(x)
            est_freq.append(y)
            est_power.append(z)
        est_dist = np.array(est_dist)
        est_freq = np.array(est_freq)
        est_power = np.array(est_power)
        return main_data, est_dist, est_freq, est_power, groups, peaks
    
        
    def determine_target_dist(self, est_dist, est_freq, est_power, groups):
        max_power = np.max(est_power)
        max_index = np.argmax(est_power)
        threshold = max_power - 3
        weighted_sum_dist = 0
        weighted_sum_freq = 0
        num_power = 0

        for i in range(0,len(est_power)):
            if est_power[i] > threshold:
                weighted_sum_dist += (est_power[i] * est_dist[i])
                weighted_sum_freq += (est_power[i] * est_freq[i])
                num_power += est_power[i]
        dist = weighted_sum_dist / num_power
        freq = weighted_sum_freq / num_power
        
        self.signal_freq = freq
        self.range_estimate = dist
        self.freq_index = max_index
        
        return freq, dist, max_index
        
    
    def polar_to_imag(self, mag, phi):
        imag = cmath.rect(mag, phi)     # phi in radians
        return imag
    
    
    def control_phaser_beam(self, my_phaser, sig_freq, control_beam = 'Full', steer_angle = None):
        
        if steer_angle == None:
            phase_delta = (
                    2
                    * 3.14159
                    * sig_freq
                    * 0.014
                    * np.sin(np.radians(self.angle_estimate))
                    / (3e8)
                    )
            phase_delta = np.degrees(phase_delta)
        else:
            phase_delta = (
                    2
                    * 3.14159
                    * sig_freq
                    * 0.014
                    * np.sin(np.radians(steer_angle))
                    / (3e8)
                    )
            phase_delta = np.degrees(phase_delta)
        
        match control_beam:
            case 'Beam1':
                channel_list = [5, 6, 7, 8]
                for ch in range(0, 4):
                    my_phaser.elements.get(channel_list[ch]).rx_phase = (
                        ((np.rint(phase_delta * ch / my_phaser.phase_step_size)) * my_phaser.phase_step_size)
                        + my_phaser.pcal[ch]
                    ) % 360.0

                my_phaser.latch_rx_settings()
                time.sleep(0.1)
                return my_phaser
                
            case 'Beam0':
                channel_list = [1, 2, 3, 4]
                for ch in range(0, 4):
                    my_phaser.elements.get(channel_list[ch]).rx_phase = (
                        ((np.rint(phase_delta * ch / my_phaser.phase_step_size)) * my_phaser.phase_step_size)
                        + my_phaser.pcal[ch]
                    ) % 360.0

                my_phaser.latch_rx_settings()
                time.sleep(0.1)
                return my_phaser
                
            case 'Full':
                #my_phaser.set_beam_phase_diff(np.degrees(phase_delta))
                channel_list = [1, 2, 3, 4, 5, 6, 7, 8]
                for ch in range(0, 8):
                    my_phaser.elements.get(channel_list[ch]).rx_phase = (
                        ((np.rint(phase_delta * ch / my_phaser.phase_step_size)) * my_phaser.phase_step_size)
                        + my_phaser.pcal[ch]
                    ) % 360.0

                my_phaser.latch_rx_settings()
                time.sleep(0.1)
                return my_phaser
                
            case _:
                print("Select Beam Control Method")
                return False
            
            
    def generate_filter(self, nSamples, sample_rate):
        Bandwidth = 2e3
        pass_band = [self.signal_freq - Bandwidth/2, self.signal_freq + Bandwidth/2]
        self.signal_filter = scp.signal.firwin(nSamples, pass_band, pass_zero=False, fs=sample_rate)
        return self.signal_filter
    
    def apply_filter(self, signal_filter, data):
        data_cuda = cp.asarray(data)
        filter_cuda = cp.asarray(signal_filter)
        filtered_sig = cupyx.scipy.signal.fftconvolve(data_cuda, filter_cuda, mode='same')
        filtered_sig = cp.asnumpy(filtered_sig)
        return self.filtered_sig
    
    def update_power_threshold(self):
        """
        Updates the power threshold after the new power is allocated

        """
        self.power_threshold = self.current_power - 10
        
        
    def monitor_target_power(self):
        """
        Monitor target power and update tracking flags
        """
        # Power at target frequency
        current_power = self.detected_power
        
        #self.current_power = current_power
        self.power_history.append(current_power)
        
        # Keep only last 10 power measurements
        if len(self.power_history) > 10:
            self.power_history.pop(0)
        
        # Check if power dropped below threshold
        if current_power < self.power_threshold:
            self.consecutive_loss_count += 1
            if self.consecutive_loss_count >= 2:  # Confirm loss with 2 consecutive measurements
                self.target_lost_flag = True
                print(f"Target {self.target_number}: Power loss detected ({current_power:.1f} dBFS)")
        else:
            self.consecutive_loss_count = 0
            self.target_lost_flag = False
        
        return current_power

    def check_frequency_deviation(self, freq):
        """
        Check if target frequency has deviated beyond threshold
        """
        if len(self.signal_freq) > 0:
            current_freq = self.signal_freq # Take detected frequency
            freq_deviation = abs(current_freq - freq)
            
            if freq_deviation > self.frequency_deviation_threshold:
                self.frequency_shifted = True
                print(f"Target {self.target_number}: Frequency shift detected ({freq_deviation/1000:.1f} kHz)")
                return True
            else:
                self.frequency_shifted = False
                return False
        return False, self.last_valid_freq

    def increase_filter_bandwidth(self, nSamples, sample_rate, bandwidth_multiplier=2):
        """
        Increase filter bandwidth when frequency shift is detected
        """
        new_bandwidth = min(5e3 * bandwidth_multiplier, self.max_filter_bandwidth)
        pass_band = [self.signal_freq - new_bandwidth/2, self.signal_freq + new_bandwidth/2]
        
        # Ensure pass_band is within valid range
        pass_band[0] = max(pass_band[0], 1000)  # Minimum 1kHz
        pass_band[1] = min(pass_band[1], sample_rate/2 - 1000)  # Below Nyquist
        
        self.signal_filter = scp.signal.firwin(nSamples, pass_band, pass_zero=False, fs=sample_rate)
        print(f"Target {self.target_number}: Filter bandwidth increased to {new_bandwidth/1000:.1f} kHz")
        return self.signal_filter


    def update_target_data(self, raw_data, Freqs, Indices, dist):
        """
        Tracking update method that handles flag-based operations
        """
        # Process raw data
        raw_data = cp.asarray(raw_data)
        filtered_sig = cupyx.scipy.signal.fftconvolve(raw_data, cp.asarray(self.signal_filter), mode='same')
        filtered_sig = cp.asnumpy(filtered_sig)
        
        #data_freq_domain = cupyx.scipy.signal.fft(filtered_sig)
        #data_freq_domain = np.abs(cp.asnumpy(data_freq_domain))[Indices]
        raw_data = np.array(self.dbfs(filtered_sig))
        data_freq_domain = raw_data[Indices]
        
        max_index = np.argmax(data_freq_domain)
        max_power = np.max(data_freq_domain)
        self.detected_power = max_power
        self.detected_freq = Freqs[max_index]
        
        return self.detected_power
    
    def update_new_data(self, freq, angle, slope):
        """
        Method to update the new object data
        """
        c = 3e8
        
        self.angle_estimate = angle
        self.signal_freq = freq
    
        dist = (freq - 100e3) * c / (4 * slope)
        self.range_estimate = dist
        
        return True
    
    
class PhasedArrayDualBeam():
    
    def __init__(self, frequency: float = 10.5e9, N_elements: int = 8, element_spacing: float = None):
        """
        Initialize phased array parameters

        Parameters:
        -----------
        frequency : float
            Operating frequency in Hz (default: 10.5 GHz)
        N_elements : int
            Number of array elements (default: 8)
        element_spacing : float
            Element spacing in meters (default: lambda/2)
        """
        self.frequency = frequency
        self.N_elements = N_elements
        self.c = 3e8  # speed of light
        self.wavelength = self.c / frequency

        if element_spacing is None:
            self.element_spacing = self.wavelength / 2
        else:
            self.element_spacing = element_spacing

        # Array geometry - linear array centered at origin
        self.element_positions = np.arange(N_elements) * self.element_spacing
        self.element_positions = self.element_positions - np.mean(self.element_positions)

        self.k = 2 * np.pi / self.wavelength  # wave number

    def steering_vector(self, angle_deg: float) -> np.ndarray:
        """
        Calculate steering vector for a given angle

        Parameters:
        -----------
        angle_deg : float
            Steering angle in degrees (broadside = 0Â°)

        Returns:
        --------
        np.ndarray
            Complex steering vector
        """
        angle_rad = np.deg2rad(angle_deg)
        phase_shifts = self.k * self.element_positions * np.sin(angle_rad)
        return np.exp(1j * phase_shifts)

    def array_factor(self, weights: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
        """
        Calculate array factor for given weights and angles

        Parameters:
        -----------
        weights : np.ndarray
            Complex weights for each element
        angles_deg : np.ndarray
            Array of angles to evaluate (degrees)

        Returns:
        --------
        np.ndarray
            Array factor values
        """
        angles_rad = np.deg2rad(angles_deg)
        AF = np.zeros(len(angles_rad), dtype=complex)

        for i, angle in enumerate(angles_rad):
            phase_shifts = self.k * self.element_positions * np.sin(angle)
            steering_vec = np.exp(1j * phase_shifts)
            AF[i] = np.sum(weights * steering_vec)

        return AF

    def calculate_dual_beam_weights(self, target_angles: List[float], 
                                  method: str = 'least_squares') -> Dict:
        """
        Calculate magnitude and phase weights for dual beam formation

        Parameters:
        -----------
        target_angles : List[float]
            Two target beam directions in degrees
        method : str
            Beamforming method: 'simple', 'least_squares', or 'mvdr'

        Returns:
        --------
        Dict
            Dictionary containing weights and array information
        """
        # Input validation
        if len(target_angles) != 2:
            raise ValueError("Exactly 2 target angles must be provided")

        if not all(-90 <= angle <= 90 for angle in target_angles):
            raise ValueError("Target angles must be between -90Â° and +90Â°")

        if target_angles[0] == target_angles[1]:
            raise ValueError("Target angles must be different")

        if method not in ['simple', 'least_squares', 'mvdr']:
            raise ValueError("Method must be 'simple', 'least_squares', or 'mvdr'")

        # Calculate weights using specified method
        if method == 'simple':
            weights = self._simple_superposition(target_angles)
        elif method == 'least_squares':
            weights = self._least_squares_optimization(target_angles)
        elif method == 'mvdr':
            weights = self._mvdr_optimization(target_angles)

        # Extract magnitude and phase
        magnitude = np.abs(weights)
        phase_radians = np.angle(weights)
        phase_degrees = np.rad2deg(phase_radians)

        # Create result dictionary
        result = {
            'magnitude': magnitude,
            'phase_degrees': phase_degrees,
            'phase_radians': phase_radians,
            'complex': weights,
            'target_angles': target_angles,
            'method': method,
            'array_info': {
                'frequency_ghz': self.frequency / 1e9,
                'wavelength_mm': self.wavelength * 1000,
                'N_elements': self.N_elements,
                'element_spacing_mm': self.element_spacing * 1000,
                'element_positions_mm': self.element_positions * 1000
            }
        }

        return result

    def _simple_superposition(self, target_angles: List[float]) -> np.ndarray:
        """Simple superposition of steering vectors"""
        steer_vec1 = self.steering_vector(target_angles[0])
        steer_vec2 = self.steering_vector(target_angles[1])
        weights = (steer_vec1 + steer_vec2) / 2
        weights = weights / np.sum(np.abs(weights)) * self.N_elements
        return weights

    def _least_squares_optimization(self, target_angles: List[float]) -> np.ndarray:
        """Least squares optimization for dual beams"""
        # Build constraint matrix
        A = np.zeros((2, self.N_elements), dtype=complex)
        A[0, :] = self.steering_vector(target_angles[0])
        A[1, :] = self.steering_vector(target_angles[1])

        # Desired response vector
        b = np.array([1.0, 1.0], dtype=complex)

        # Solve using pseudoinverse
        weights = np.linalg.pinv(A) @ b
        return weights

    def _mvdr_optimization(self, target_angles: List[float]) -> np.ndarray:
        """MVDR optimization for dual beams"""
        # Build constraint matrix
        A = np.zeros((2, self.N_elements), dtype=complex)
        A[0, :] = self.steering_vector(target_angles[0])
        A[1, :] = self.steering_vector(target_angles[1])

        R = np.eye(self.N_elements, dtype=complex)

        # Add some correlation between adjacent elements
        correlation = 0.2
        for i in range(self.N_elements - 1):
            R[i, i+1] = correlation
            R[i+1, i] = correlation

        # Add some thermal noise variance scaling
        noise_scaling = np.linspace(0.8, 1.2, self.N_elements)
        for i in range(self.N_elements):
            R[i, i] *= noise_scaling[i]

        f = np.array([1.0, 1.0], dtype=complex)

        # MVDR solution
        R_inv = np.linalg.inv(R)
        AH = A.conj().T
        temp = A @ R_inv @ AH
        weights = R_inv @ AH @ np.linalg.inv(temp) @ f
        return weights

    def validate_pattern(self, weights_result: Dict, plot_angles: np.ndarray = None) -> Dict:
        """
        Validate the beam pattern and calculate performance metrics

        Parameters:
        -----------
        weights_result : Dict
            Output from calculate_dual_beam_weights
        plot_angles : np.ndarray
            Angles for pattern evaluation

        Returns:
        --------
        Dict
            Validation results including pattern data and metrics
        """
        if plot_angles is None:
            plot_angles = np.linspace(-90, 90, 361)

        # Calculate array factor
        AF = self.array_factor(weights_result['complex'], plot_angles)
        AF_db = 20 * np.log10(np.abs(AF) + 1e-12)
        AF_db = AF_db - np.max(AF_db)  # Normalize to 0 dB peak

        # Find peaks
        peaks = []
        for i in range(1, len(AF_db)-1):
            if AF_db[i] > AF_db[i-1] and AF_db[i] > AF_db[i+1] and AF_db[i] > -10:
                peaks.append(i)

        peak_angles = plot_angles[peaks] if peaks else np.array([])
        peak_levels = AF_db[peaks] if peaks else np.array([])

        # Calculate target responses
        target_responses = []
        for angle in weights_result['target_angles']:
            idx = np.argmin(np.abs(plot_angles - angle))
            target_responses.append(AF_db[idx])

        # Calculate pointing errors
        pointing_errors = []
        if len(peak_angles) >= 2:
            for target in weights_result['target_angles']:
                closest_peak_idx = np.argmin(np.abs(peak_angles - target))
                error = abs(peak_angles[closest_peak_idx] - target)
                pointing_errors.append(error)

        return {
            'angles': plot_angles,
            'pattern_db': AF_db,
            'pattern_linear': np.abs(AF),
            'peak_angles': peak_angles,
            'peak_levels': peak_levels,
            'target_responses': target_responses,
            'pointing_errors': pointing_errors,
            'avg_pointing_error': np.mean(pointing_errors) if pointing_errors else None,
            'sidelobe_level': np.mean(np.sort(AF_db)[:50])  # Average of lowest values
        }

    def plot_beam_pattern(self, weights_result: Dict, figsize: Tuple[int, int] = (10, 6)):
        """
            Plot the beam pattern

            Parameters:
                -----------
                weights_result : Dict
                Output from calculate_dual_beam_weights
                figsize : Tuple[int, int]
                Figure size
        """
        validation = self.validate_pattern(weights_result)

        plt.figure(figsize=figsize)
        plt.plot(validation['angles'], validation['pattern_db'], 'b-', linewidth=2)
        plt.axhline(y=-3, color='r', linestyle='--', alpha=0.5, label='-3 dB')

        # Mark target angles
        for angle in weights_result['target_angles']:
            plt.axvline(x=angle, color='g', linestyle=':', alpha=0.7, 
                       label=f'Target: {angle}Â°' if angle == weights_result['target_angles'][0] 
                       else f'{angle}Â°')

        plt.xlabel('Angle (degrees)')
        plt.ylabel('Normalized Pattern (dB)')
        plt.title(f'Dual Beam Pattern - Method: {weights_result["method"]}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim([-90, 90])
        plt.ylim([-40, 5])
        plt.show()


# Convenience function for direct use
def phased_array_dual_beam_weights(target_angles: List[float], 
                                  frequency: float = 10.5e9, 
                                  N_elements: int = 8, 
                                  element_spacing: float = None,
                                  method: str = 'least_squares') -> Dict:
    """
        Calculate magnitude and phase weights for dual beam phased array

        Parameters:
            -----------
            target_angles : List[float]
            Two target beam directions in degrees
            frequency : float
            Operating frequency in Hz (default: 10.5 GHz)
            N_elements : int
            Number of array elements (default: 8)
            element_spacing : float
            Element spacing in meters (default: lambda/2)
        method : str
            Beamforming method (default: 'least_squares')

        Returns:
            --------
            Dict
                Dictionary with magnitude, phase weights and array info

    """
    array = PhasedArrayDualBeam(frequency, N_elements, element_spacing)
    return array.calculate_dual_beam_weights(target_angles, method)

