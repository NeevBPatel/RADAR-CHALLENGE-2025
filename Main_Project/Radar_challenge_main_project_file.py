import numpy as np
import scipy as scp
import pickle
import matplotlib.pyplot as plt
import adi
import os
import time
from target_detection_dbfs import cfar
import helpers
from scipy.signal.windows import taylor
from custom_classes import target, PhasedArrayDualBeam, phased_array_dual_beam_weights
import cupy as cp
import cupyx.scipy.signal
import cmath
import signal
import sys

print("All libraries imported!!!")

master_flag = False
flag_0 = False
flag_1 = False
continuous_operation = True  # Flag for continuous operation

sig_freq = 10.045e9  #10.25e9
sample_rate = 1e6
nSamples = int(2**13)
tx_gain = -0
rx_gain = 20
sample_rate = 1e6
center_freq = 2.1e9
data_signal_freq = 100e3

sdr_ip = "ip:192.168.2.1"
rpi_ip = "ip:phaser.local"

#rx_gain_list = [25, 118, 97, 119, 122, 114, 66, 42]
rx_gain_list = [127]*8

# Signal handler for shutdown
def signal_handler(sig, frame):
    global continuous_operation
    print('\n\nReceived Ctrl+C! Shutting down...')
    continuous_operation = False
    print('Cleaning up...')

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

#%% Initiate Devices
( my_phaser, my_sdr, iq, Freqs, Indices, dist, slope ) = helpers.initiate_all_devices(
    sdr_ip, rpi_ip, sample_rate, center_freq, nSamples, data_signal_freq, rx_gain, rx_gain_list
)

#%% Extra Functions
def get_target_update(target_i, data):
    raw_data = data
    target_i.update_target_data(raw_data, Freqs, Indices, dist)

def monitor_targets(my_phaser, my_sdr, targets):
    global flag_0
    global flag_1
    global master_flag
    
    data = my_sdr.rx()
    data = data[0] + data[1]
    
    get_target_update(targets[0], data)
    targets[0].monitor_target_power()
    flag_0 = targets[0].target_lost_flag
    
    get_target_update(targets[1], data)
    targets[1].monitor_target_power()
    flag_1 = targets[1].target_lost_flag
    
    master_flag = flag_0 or flag_1
    print(f"Target Status - T0: {'LOST' if flag_0 else 'OK'}, T1: {'LOST' if flag_1 else 'OK'}")

def generate_filter_external(signal_freq, nSamples, sample_rate):
    Bandwidth = 2e3
    pass_band = [signal_freq - Bandwidth/2, signal_freq + Bandwidth/2]
    signal_filter = scp.signal.firwin(nSamples, pass_band, pass_zero=False, fs=sample_rate)
    return signal_filter

def apply_filter_external(signal_filter, data):
    data_cuda = cp.asarray(data)
    filter_cuda = cp.asarray(signal_filter)
    filtered_sig = cupyx.scipy.signal.fftconvolve(data_cuda, filter_cuda, mode='same')
    filtered_sig = cp.asnumpy(filtered_sig)
    return filtered_sig

def analyze_data_matrix(data_matrix, reacquire_freq, reacquire_angles):
    max_index = np.where(data_matrix==data_matrix.max())
    temp = max_index[0][0]
    new_frew_index = int(temp)
    temp = max_index[1][0]
    new_angle_index = int(temp)
    new_freq = reacquire_freq[new_frew_index]
    new_angle = reacquire_angles[new_angle_index]
    return max_index, new_freq, new_angle

def perform_full_beam_scan():
    """
    Perform a complete beam scan to reacquire both targets when both are lost
    Similar to the initial scan but more comprehensive for reacquisition
    """
    print("Performing full beam scan for dual target reacquisition...")
    
    angles = np.linspace(-70, 70, 141)  # Same as initial scan
    powers = []
    main_filter = scp.signal.firwin(nSamples, [110000, 400000], pass_zero=False, fs=sample_rate)
    
    for i in range(0, len(angles)):
        phase_delta = (
            2
            * 3.14159
            * sig_freq
            * 0.014
            * np.sin(np.radians(angles[i]))
            / (3e8)
        )
        
        my_phaser.set_beam_phase_diff(np.degrees(phase_delta))
        data = my_sdr.rx()
        sum_data = data[0] + data[1]
        sum_data = apply_filter_external(main_filter, sum_data)
        power_dB = 10*np.log10(np.sum(np.abs(sum_data)**2))
        powers.append(power_dB)
    
    powers = np.array(powers)
    
    # Apply edge suppression
    for i in range(0, 5):
        powers[i] = powers[-i] = np.max(powers) - 30
    
    # CFAR detection
    cfar_method = 'greatest'
    num_guard_cells = 1
    num_ref_cells = 28
    bias = 3
    peaks, targets_detected = cfar(powers, num_guard_cells, num_ref_cells, bias, cfar_method)
    est = np.invert(targets_detected.mask)
    num_groups, groups = helpers.find_groups_of_consecutive_false(est)
    
    est_angles = []
    for i in range(0, num_groups):
        indices = groups[i]
        x = np.mean(angles[indices[0]:indices[1]])
        est_angles.append(x)
    
    est_angles = np.array(est_angles)
    
    print(f"Full beam scan completed. Found {len(est_angles)} potential targets at angles: {est_angles}")
    
    return est_angles, angles, powers, peaks

def reacquire_both_targets(targets, my_phaser):
    """
    Reacquisition strategy when both targets are lost
    Performs full beam scan and matches detected targets to existing target objects
    """
    
    # Store original target parameters for reference
    original_angles = [targets[0].angle_estimate, targets[1].angle_estimate]
    original_freqs = [targets[0].signal_freq, targets[1].signal_freq]
    original_ranges = [targets[0].range_estimate, targets[1].range_estimate]
    
    print(f"Original target parameters:")
    print(f"  Target 0: Angle={original_angles[0]:.1f}°, Freq={original_freqs[0]/1000:.1f}kHz, Range={original_ranges[0]:.1f}m")
    print(f"  Target 1: Angle={original_angles[1]:.1f}°, Freq={original_freqs[1]/1000:.1f}kHz, Range={original_ranges[1]:.1f}m")
    
    # Perform full beam scan
    est_angles, scan_angles, scan_powers, cfar_threshold = perform_full_beam_scan()
    
    if len(est_angles) >= 2:
        # Sort detected angles and match to closest original targets
        est_angles_sorted = np.sort(est_angles)
        
        # Take the two strongest or most likely candidates
        if len(est_angles_sorted) >= 2:
            candidate_angles = est_angles_sorted[:2]  # Take first two
        else:
            candidate_angles = est_angles_sorted
            
        print(f"Using candidate angles: {candidate_angles}")
        
        # Match candidates to original targets based on angular proximity
        target_assignments = []
        for original_index, original_angle in enumerate(original_angles):
            distances = [abs(cand_angle - original_angle) for cand_angle in candidate_angles]
            best_candidate_idx = np.argmin(distances)
            target_assignments.append((original_index, candidate_angles[best_candidate_idx]))
        
        # Update targets with new angles and perform detailed reacquisition
        for target_index, new_angle in target_assignments:
            print(f"Reacquiring Target {target_index} at new angle: {new_angle:.1f}°")
            
            # Point beam to new angle
            targets[target_index].angle_estimate = new_angle
            my_phaser = targets[target_index].control_phaser_beam(my_phaser, sig_freq)
            
            # Perform frequency and range analysis similar to individual target recovery
            initial_range = original_ranges[target_index]
            initial_target_freq = original_freqs[target_index]
            
            # Generate frequency candidates based on original range
            if initial_range <= 2:
                freq_dev = np.arange(0, 17000, 2000)
                reacquire_freq = freq_dev + initial_target_freq
            else:
                freq_dev = np.arange(-8000, 10000, 2000)
                reacquire_freq = freq_dev + initial_target_freq
            
            # Test different frequencies at the new angle
            best_power = -np.inf
            best_freq = initial_target_freq
            
            for test_freq in reacquire_freq:
                freq_filter = generate_filter_external(test_freq, nSamples, sample_rate)
                raw_data = my_sdr.rx()
                raw_data = raw_data[0] + raw_data[1]
                filtered_sig = apply_filter_external(freq_filter, raw_data)
                power_dB = 10*np.log10(np.sum(np.abs(filtered_sig)**2))
                
                if power_dB > best_power:
                    best_power = power_dB
                    best_freq = test_freq
            
            # Update target with best parameters
            targets[target_index].update_new_data(best_freq, new_angle, slope)
            targets[target_index].generate_filter(nSamples, sample_rate)
            
            print(f"Target {target_index} reacquired: Angle={new_angle:.1f}°, Freq={best_freq/1000:.1f}kHz, Power={best_power:.1f}dB")
        
        return True
        
    elif len(est_angles) == 1:
        print("Only one target detected in beam scan. Attempting proximity search for second target...")
        
        # Assign the one detected target to the closest original target
        detected_angle = est_angles[0]
        distances_to_originals = [abs(detected_angle - orig_angle) for orig_angle in original_angles]
        closest_target_idx = np.argmin(distances_to_originals)
        
        # Update the closest target
        targets[closest_target_idx].angle_estimate = detected_angle
        targets[closest_target_idx].update_new_data(original_freqs[closest_target_idx], detected_angle, slope)
        targets[closest_target_idx].generate_filter(nSamples, sample_rate)
        
        # Search for the other target around its original position
        other_target_idx = 1 - closest_target_idx
        found_second = search_for_missing_target(targets[other_target_idx], original_angles[other_target_idx], my_phaser)
        
        if found_second:
            print("Both targets successfully reacquired!")
            return True
        else:
            print("Could only reacquire one target. Continuing search...")
            return False
            
    else:
        print("No targets detected in full beam scan. Expanding search parameters...")
        return False

def search_for_missing_target(target_obj, search_center_angle, my_phaser):
    """
    Intensive search for a missing target around a center angle
    Used when full beam scan doesn't find both targets
    """
    print(f"Searching for missing target around {search_center_angle:.1f}°...")
    
    # Expanded search parameters
    search_range = 20  # degrees
    angle_step = 1     # degree resolution
    
    search_angles = np.arange(search_center_angle - search_range/2, 
                             search_center_angle + search_range/2, 
                             angle_step)
    
    # Ensure angles are within valid range
    search_angles = search_angles[search_angles >= -70]
    search_angles = search_angles[search_angles <= 70]
    
    best_power = -np.inf
    best_angle = search_center_angle
    best_freq = target_obj.signal_freq
    
    # Test multiple frequencies at each angle
    freq_range = np.arange(target_obj.signal_freq - 10000, target_obj.signal_freq + 10000, 2000)
    
    for angle in search_angles:
        for test_freq in freq_range:
            # Point beam to test angle
            my_phaser = target_obj.control_phaser_beam(my_phaser, sig_freq, control_beam='Full', steer_angle=angle)
            
            # Generate and apply frequency filter
            freq_filter = generate_filter_external(test_freq, nSamples, sample_rate)
            raw_data = my_sdr.rx()
            raw_data = raw_data[0] + raw_data[1]
            filtered_sig = apply_filter_external(freq_filter, raw_data)
            power_dB = 10*np.log10(np.sum(np.abs(filtered_sig)**2))
            
            if power_dB > best_power:
                best_power = best_power
                best_angle = angle
                best_freq = test_freq
    
    # Check if we found a significant signal
    if best_power > target_obj.power_threshold + 5:  # 5dB margin
        target_obj.update_new_data(best_freq, best_angle, slope)
        target_obj.generate_filter(nSamples, sample_rate)
        print(f"Missing target found at angle={best_angle:.1f}°, freq={best_freq/1000:.1f}kHz, power={best_power:.1f}dB")
        return True
    else:
        print(f"Missing target not found. Best signal: {best_power:.1f}dB at {best_angle:.1f}°")
        return False

#%% Scan + Detect 
angles = np.linspace(-70, 70, 141)
powers = []
main_filter = scp.signal.firwin(nSamples, [110000, 400000], pass_zero=False, fs=sample_rate)

for i in range(0,len(angles)):
    phase_delta = (
        2
        * 3.14159
        * sig_freq
        * 0.014
        * np.sin(np.radians(angles[i]))
        / (3e8)
    )
    
    my_phaser.set_beam_phase_diff(np.degrees(phase_delta))
    data = my_sdr.rx()
    sum_data = data[0] + data[1]
    sum_data = apply_filter_external(main_filter*10, sum_data)
    power_dB = 10*np.log10(np.sum(np.abs(sum_data)**2))
    powers.append(power_dB)

# Apply edge suppression
powers = np.array(powers)
for i in range(0,7):
    powers[i] = powers[-i] = np.max(powers)-30

cfar_method = 'greatest'
num_guard_cells = 1
num_ref_cells = 25
bias = 4

peaks, targets_detected = cfar(powers,num_guard_cells, num_ref_cells, bias, cfar_method)
est = np.invert(targets_detected.mask)
num_groups, groups = helpers.find_groups_of_consecutive_false(est)

est_angles = []
for i in range(0,num_groups):
    indices = groups[i]
    x = np.mean(angles[indices[0]:indices[1]])
    est_angles.append(x)

est_angles = np.array(est_angles)

plt.figure(figsize=(12, 6))
plt.plot(angles, powers, '.-', label='Received Power')
plt.plot(angles, peaks, label='CFAR Threshold')
for angle in est_angles:
    plt.axvline(x=angle, color='red', linestyle='--', alpha=0.7,
                label=f'Target: {angle:.1f}°')
plt.xlabel('Angle of Arrival (degrees)')
plt.ylabel('Power (dB)')
plt.title('Initial Beam Scan Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%% Allocate Targets 
targets = []
for i in range(0,len(est_angles)):
    est_angle = est_angles[i]
    targets.append(target(number=i,angle=est_angle))
targets = np.array(targets)

#%% Initialize target data 
for target_i in targets:
    my_phaser = target_i.control_phaser_beam(my_phaser, sig_freq)
    data = my_sdr.rx()
    raw_data = data[0] + data[1]
    data_dbfs = target_i.dbfs_cupy(raw_data)
    main_data, est_dist, est_freq, est_power, groups, spec_threshold = target_i.process_target_data(data_dbfs, Indices, Freqs, dist, 5, 150, 20, 'average')
    
    plt.plot(dist, main_data)
    plt.plot(dist, spec_threshold)
    plt.show()
    
    target_i.determine_target_dist(est_dist, est_freq, est_power, groups)
    target_i.generate_filter(nSamples, sample_rate)

#%% Form dual beam 
def form_dual_beams(angle_list, my_phaser):
    global sig_freq
    weights_raw = phased_array_dual_beam_weights(angle_list, sig_freq, 8, None)
    weights = np.array(weights_raw.get('complex'))
    weights_mag = np.abs(weights.conj())
    weights_degrees = np.rad2deg(np.angle(weights.conj()))
    weights_mag = weights_mag * 127
    max_mag = np.max(weights_mag)
    weights_mag = np.rint((weights_mag/max_mag)*127)
    
    for i in range(0, 8):
        my_phaser.set_chan_phase(i, weights_degrees[i], apply_cal=True)
    for i in range(0, 8):
        my_phaser.set_chan_gain(i, weights_mag[i], apply_cal=True)
    my_phaser.latch_rx_settings()
    
    return (weights_raw, weights_mag, weights_degrees, my_phaser)

target_1 = targets[0]
target_2 = targets[1]
ang1 = target_1.angle_estimate
ang2 = target_2.angle_estimate
angle_list = [ang1, ang2]

(weights_raw, weights_mag, weights_degrees, my_phaser) = form_dual_beams(angle_list, my_phaser)

#%% Monitor

print("Press Ctrl+C to stop \n")

iteration_count = 0
recovery_attempts = 0
max_recovery_attempts = 3

while continuous_operation:
    try:
        iteration_count += 1
        print(f"\n--- Monitoring Cycle {iteration_count} ---")
        
        # Monitor targets
        monitor_targets(my_phaser, my_sdr, targets)
        flag_array = [flag_0, flag_1]
        
        # Handle different target loss scenarios
        match flag_array:
            case [False, False]:
                # Both targets are being tracked successfully
                print("Both targets tracking normally")
                recovery_attempts = 0  # Reset recovery counter on success
                
                # Periodically reform dual beam to maintain optimal tracking
                if iteration_count % 25 == 0:  # Every 25 iterations
                    print("Refreshing dual beam configuration...")
                    current_angles = [targets[0].angle_estimate, targets[1].angle_estimate]
                    (weights_raw, weights_mag, weights_degrees, my_phaser) = form_dual_beams(current_angles, my_phaser)
            
            case [False, True]:
                # Target 1 lost, Target 0 active 
                print("Analysing lost target 1")
                recovery_attempts += 1
                
                deviation = 10
                initial_angle = targets[1].angle_estimate
                angle_dev = np.arange(-deviation, deviation + 2, 2)
                reacquire_angles = []
                for i in range(0,len(angle_dev)):
                    reacquire_angles.append(initial_angle + angle_dev[i])
                reacquire_angles = np.array(reacquire_angles)
                
                initial_range = targets[1].range_estimate
                initial_target_freq = targets[1].signal_freq
                reacquire_freq = []
                if initial_range <= 2:
                    freq_dev = np.arange(0, 17000, 2000)
                    reacquire_freq = freq_dev + initial_target_freq
                if initial_range > 2:
                    freq_dev = np.arange(-8000, 10000, 2000)
                    reacquire_freq = freq_dev + initial_target_freq
                
                data_matrix = np.zeros([len(reacquire_freq), len(reacquire_angles)],dtype=float)
                
                for n in range(0,len(reacquire_freq)):
                    analyze_freq = reacquire_freq[n]
                    freq_filter = generate_filter_external(analyze_freq, nSamples, sample_rate)
                    for m in range(0,len(reacquire_angles)):
                        my_phaser = targets[1].control_phaser_beam(my_phaser, sig_freq, control_beam = 'Full', steer_angle = reacquire_angles[m])
                        raw_data = my_sdr.rx()
                        raw_data = raw_data[0] + raw_data[1]
                        filtered_sig = apply_filter_external(freq_filter, raw_data)
                        power_dB = 10*np.log10(np.sum(np.abs(filtered_sig)**2))
                        data_matrix[n,m] = power_dB
                
                max_index, new_freq, new_angle = analyze_data_matrix(data_matrix, reacquire_freq, reacquire_angles)
                targets[1].update_new_data(new_freq, new_angle, slope)
                targets[1].generate_filter(nSamples, sample_rate)
                
                # Reform dual beam with updated target positions
                updated_angles = [targets[0].angle_estimate, targets[1].angle_estimate]
                (weights_raw, weights_mag, weights_degrees, my_phaser) = form_dual_beams(updated_angles, my_phaser)
                print(f"Target 1 recovered at angle={new_angle:.1f}°, freq={new_freq/1000:.1f}kHz")
            
            case [True, False]:
                # Target 0 lost, Target 1 active
                print("Analysing lost target 0")
                recovery_attempts += 1
                
                deviation = 10
                initial_angle = targets[0].angle_estimate
                angle_dev = np.arange(-deviation, deviation + 2, 2)
                reacquire_angles = []
                for i in range(0,len(angle_dev)):
                    reacquire_angles.append(initial_angle + angle_dev[i])
                reacquire_angles = np.array(reacquire_angles)
                
                initial_range = targets[0].range_estimate
                initial_target_freq = targets[0].signal_freq
                reacquire_freq = []
                if initial_range <= 2:
                    freq_dev = np.arange(0, 17000, 2000)
                    reacquire_freq = freq_dev + initial_target_freq
                if initial_range > 2:
                    freq_dev = np.arange(-8000, 10000, 2000)
                    reacquire_freq = freq_dev + initial_target_freq
                
                data_matrix = np.zeros([len(reacquire_freq), len(reacquire_angles)],dtype=float)
                
                for n in range(0,len(reacquire_freq)):
                    analyze_freq = reacquire_freq[n]
                    freq_filter = generate_filter_external(analyze_freq, nSamples, sample_rate)
                    for m in range(0,len(reacquire_angles)):
                        my_phaser = targets[0].control_phaser_beam(my_phaser, sig_freq, control_beam = 'Full', steer_angle = reacquire_angles[m])
                        raw_data = my_sdr.rx()
                        raw_data = raw_data[0] + raw_data[1]
                        filtered_sig = apply_filter_external(freq_filter, raw_data)
                        power_dB = 10*np.log10(np.sum(np.abs(filtered_sig)**2))
                        data_matrix[n,m] = power_dB
                
                max_index, new_freq, new_angle = analyze_data_matrix(data_matrix, reacquire_freq, reacquire_angles)
                targets[0].update_new_data(new_freq, new_angle, slope)
                targets[0].generate_filter(nSamples, sample_rate)
                
                # Reform dual beam with updated target positions
                updated_angles = [targets[0].angle_estimate, targets[1].angle_estimate]
                (weights_raw, weights_mag, weights_degrees, my_phaser) = form_dual_beams(updated_angles, my_phaser)
                print(f"Target 0 recovered at angle={new_angle:.1f}°, freq={new_freq/1000:.1f}kHz")
            
            case [True, True]:
                # Both targets lost
                print("BOTH TARGETS LOST! Initiating dual target recovery...")
                recovery_attempts += 1
                
                if recovery_attempts <= max_recovery_attempts:
                    success = reacquire_both_targets(targets, my_phaser)
                    
                    if success:
                        print("Both targets successfully reacquired!")
                        # Reform dual beam with new target positions
                        recovered_angles = [targets[0].angle_estimate, targets[1].angle_estimate]
                        (weights_raw, weights_mag, weights_degrees, my_phaser) = form_dual_beams(recovered_angles, my_phaser)
                        recovery_attempts = 0  # Reset counter on success
                    else:
                        print(f"Dual recovery attempt {recovery_attempts} failed. Retrying...")
                        time.sleep(1)  # Brief pause before retry
                else:
                    print(f"Maximum recovery attempts ({max_recovery_attempts}) exceeded.")
                    print("Consider manual intervention or restart.")
                    recovery_attempts = 0  # Reset for next loss event
            
            case _:
                print(f"Unknown flag combination: {flag_array}")
        
        # Brief pause between monitoring cycles
        time.sleep(0.1)
        
        # Status update every 50 iterations
        if iteration_count % 50 == 0:
            print(f"\n=== Status Update (Cycle {iteration_count}) ===")
            print(f"Target 0: Angle={targets[0].angle_estimate:.1f}°, Range={targets[0].range_estimate:.1f}m, Power={targets[0].detected_power:.1f}dB")
            print(f"Target 1: Angle={targets[1].angle_estimate:.1f}°, Range={targets[1].range_estimate:.1f}m, Power={targets[1].detected_power:.1f}dB")
            print(f"Recovery attempts: {recovery_attempts}")
            print("=" * 40)
    
    except Exception as e:
        print(f"Error in monitoring cycle {iteration_count}: {e}")
        print("Continuing with next cycle...")
        time.sleep(0.5)

print(f"\nMonitoring stopped after {iteration_count} cycles.")
print("Performing cleanup...")

#%% TX Buffer Destroy
try:
    my_sdr.tx_destroy_buffer()
    print("SDR buffer destroyed successfully.")
except:
    print("Error destroying SDR buffer.")


print("Cleanup complete. System shutdown.")
