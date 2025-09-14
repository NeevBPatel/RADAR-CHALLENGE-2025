import numpy as np
import scipy as scp
import pickle
import matplotlib.pyplot as plt
import adi
import os
import time
from target_detection_dbfs import cfar
from scipy.signal.windows import taylor
import cupy as cp
import cupyx.scipy.signal


def initiate_all_devices(sdr_ip, rpi_ip, sample_rate, center_freq, nSamples, data_signal_freq, rx_gain, rx_gain_list):
    '''
        Initiates the phaser and Pluto SDR with all the required settings
    '''
    
    my_sdr = adi.ad9361(uri=sdr_ip)
    my_phaser = adi.CN0566(uri=rpi_ip, sdr=my_sdr)
    
    print("Phaser And PlutoSDR connected")
    
    time.sleep(0.05)

    # Initialize both ADAR1000s, set gains to max, and all phases to 0
    my_phaser.configure(device_mode="rx")
    my_phaser.load_gain_cal()
    my_phaser.load_phase_cal()
    for i in range(0, 8):
        my_phaser.set_chan_phase(i, 0)

    gain_list = rx_gain_list #[8, 34, 84, 127, 127, 84, 34, 8]  # Blackman taper
    for i in range(0, len(gain_list)):
        my_phaser.set_chan_gain(i, gain_list[i], apply_cal=True)

    # Setup Raspberry Pi GPIO states

    my_phaser._gpios.gpio_tx_sw = 0  # 0 = TX_OUT_2, 1 = TX_OUT_1
    my_phaser._gpios.gpio_vctrl_1 = 1  # 1=Use onboard PLL/LO source  (0=disable PLL and VCO, and set switch to use external LO input)
    my_phaser._gpios.gpio_vctrl_2 = (
            1  # 1=Send LO to transmit circuitry  (0=disable Tx path, and send LO to LO_OUT)
        )


    # Configure SDR Rx
    my_sdr.sample_rate = int(sample_rate)
    my_sdr.rx_lo = int(center_freq)  # set this to output_freq - (the freq of the HB100)
    my_sdr.rx_enabled_channels = [0, 1]  # enable Rx1 (voltage0) and Rx2 (voltage1)
    my_sdr.rx_buffer_size = int(nSamples)
    my_sdr.gain_control_mode_chan0 = "manual"  # manual or slow_attack
    my_sdr.gain_control_mode_chan1 = "manual"  # manual or slow_attack
    my_sdr.rx_hardwaregain_chan0 = int(rx_gain)  # must be between -3 and 70
    my_sdr.rx_hardwaregain_chan1 = int(rx_gain)  # must be between -3 and 70
    # Configure SDR Tx
    my_sdr.tx_lo = int(center_freq)
    my_sdr.tx_enabled_channels = [0, 1]
    my_sdr.tx_cyclic_buffer = True  # must set cyclic buffer to true for the tdd burst mode.  Otherwise Tx will turn on and off randomly
    my_sdr.tx_hardwaregain_chan0 = -88  # must be between 0 and -88
    my_sdr.tx_hardwaregain_chan1 = -0  # must be between 0 and -88

    # Configure the ADF4159 Rampling PLL
    output_freq = 12.145e9
    BW = 500e6
    num_steps = 1000
    ramp_time = 1e3  # us
    ramp_time_s = ramp_time / 1e6
    my_phaser.frequency = int(output_freq / 4)  # Output frequency divided by 4
    my_phaser.freq_dev_range = int(
        BW / 4
    )  # frequency deviation range in Hz.  This is the total freq deviation of the complete freq ramp
    my_phaser.freq_dev_step = int(
        (BW / 4) / num_steps
    )  # frequency deviation step in Hz.  This is fDEV, in Hz.  Can be positive or negative
    my_phaser.freq_dev_time = int(
        ramp_time
    )  # total time (in us) of the complete frequency ramp
    print("requested freq dev time = ", ramp_time)
    print("actual freq dev time = ", my_phaser.freq_dev_time)
    my_phaser.delay_word = 4095  # 12 bit delay word.  4095*PFD = 40.95 us.  For sawtooth ramps, this is also the length of the Ramp_complete signal
    my_phaser.delay_clk = "PFD"  # can be 'PFD' or 'PFD*CLK1'
    my_phaser.delay_start_en = 0  # delay start
    my_phaser.ramp_delay_en = 0  # delay between ramps.
    my_phaser.trig_delay_en = 0  # triangle delay
    my_phaser.ramp_mode = "continuous_triangular"  # ramp_mode can be:  "disabled", "continuous_sawtooth", "continuous_triangular", "single_sawtooth_burst", "single_ramp_burst"
    my_phaser.sing_ful_tri = (
        0  # full triangle enable/disable -- this is used with the single_ramp_burst mode
    )
    my_phaser.tx_trig_en = 0  # start a ramp with TXdata
    my_phaser.enable = 0  # 0 = PLL enable.  Write this last to update all the registers

    # Print config
    print(
        """
    CONFIG:
    Sample rate: {sample_rate}MHz
    Num samples: {samples}
    Bandwidth: {BW}MHz
    Ramp time: {ramp_time}ms
    Output frequency: {output_freq}MHz
    IF: {signal_freq}kHz
    """.format(
            sample_rate=sample_rate / 1e6,
            samples=int(nSamples),
            BW=BW / 1e6,
            ramp_time=ramp_time / 1e3,
            output_freq=output_freq / 1e6,
            signal_freq=data_signal_freq / 1e3,
        )
    )

    # Create a sinewave waveform
    fs = int(my_sdr.sample_rate)
    print("sample_rate:", fs)
    N = int(my_sdr.rx_buffer_size)
    fc = int(data_signal_freq / (fs / N)) * (fs / N)
    ts = 1 / float(fs)
    t = np.arange(0, N * ts, ts)
    i = np.cos(2 * np.pi * t * fc) * 2 ** 14
    q = np.sin(2 * np.pi * t * fc) * 2 ** 14
    iq = 1 * (i + 1j * q)

    # Send data
    my_sdr._ctx.set_timeout(0)
    my_sdr.tx([iq * 0.5, iq])  # only send data to the 2nd channel (that's all we need)

    c = 3e8
    default_rf_bw = 500e6
    freqs = np.fft.fftfreq(nSamples, 1/sample_rate)
    Freqs = []
    Indices = []
    slope = BW / ramp_time_s

    for i in range(0,len(freqs)):
        if freqs[i]>105e3:
            Freqs.append(freqs[i])
            Indices.append(i)
            
    Freqs = np.array(Freqs)
    Indices = np.array(Indices)

    dist = (Freqs - data_signal_freq) * c / (4 * slope)
    
    return my_phaser, my_sdr, iq, Freqs, Indices, dist, slope
    
    
def find_groups_of_consecutive_false(bool_array):
    """
    Identifies and counts groups of consecutive 'False' elements in a list of booleans.
    A group is defined as a sequence of one or more 'False' values at
    consecutive indices.

    Args:
        bool_array (list): A list of boolean values (True/False).

    Returns:
        tuple: A tuple containing:
            - int: The total number of groups of consecutive 'False' elements.
            - list: A nested list where each sublist contains the start and 
                    end indices of such a group.
    """
    
    groups = []
    in_false_group = False
    start_index = None

    # Iterate through the array with index
    for i, value in enumerate(bool_array):
        # Case 1: Start of a new group of 'False' elements
        if not value and not in_false_group:
            in_false_group = True
            start_index = i
        # Case 2: End of a group of 'False' elements (when a True is found)
        elif value and in_false_group:
            end_index = i - 1
            groups.append([start_index, end_index])
            in_false_group = False
            start_index = None # Reset for clarity

    # Edge case: If the array ends while inside a 'False' group
    if in_false_group:
        end_index = len(bool_array) - 1
        groups.append([start_index, end_index])

    # The number of groups is the length of our list of groups
    num_groups = len(groups)

    return num_groups, groups

def dbfs(raw_data):
    # function to convert IQ samples to FFT plot, scaled in dBFS
    #NumSamples = len(raw_data)
    win_funct = np.hanning(len(raw_data))
    y = raw_data * win_funct
    sp = np.absolute(np.fft.fft(y))
    #sp = np.fft.fftshift(sp)
    s_mag = np.abs(sp) / np.sum(win_funct)
    s_mag = np.maximum(s_mag, 10 ** (-15))
    s_dbfs = 20 * np.log10(s_mag / (2 ** 11))
    return  s_dbfs

def get_target_data(data_dbfs, indices, freqs, dist, guard_cells, ref_cells, bias, method):
    main_data = data_dbfs[indices]
    peaks, targets = cfar(main_data, guard_cells, ref_cells, bias, method)
    
    num_groups, groups = find_groups_of_consecutive_false(targets.mask)

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
    return ( main_data, est_dist, est_freq, est_power, groups ) 

def get_monopulse_data(my_phaser, my_sdr, Averages, index_range):
    total_sum = 0
    total_delta = 0
    total_beam_phase = 0
    if index_range[0] == index_range[1]:
        index = int(index_range[0])
    else:
        index = int(np.floor(np.mean(index_range)))
    
    for count in range(0, Averages):
        data = my_sdr.rx()
        chan1 = data[0]  # Rx1 data
        chan2 = data[1]  # Rx2 data
        
        win_funct = taylor(len(chan1),nbar=10,sll=50)
        
        # scale the amplitude from the "digital tab"
        chan1 = chan1 * 40 * win_funct
        chan2 = chan2 * 40 * win_funct
        nSamples = len(chan1)
        # shift phase from the "digital tab"
        dig_Beam0_phase = np.deg2rad(12)
        dig_Beam1_phase = np.deg2rad(-12)
        if dig_Beam0_phase != 0:
            chan1_fft = np.fft.fft(chan1)
            chan1_fft[:index-40] = -150
            chan1_fft[index+40:] = -150
            chan1_fft_shift = chan1_fft * np.exp(1.0j * dig_Beam0_phase)
            chan1 = np.fft.ifft(chan1_fft_shift, n=nSamples)
            chan1 = chan1 * win_funct
            chan1 = chan1[0:nSamples]
        if dig_Beam1_phase != 0:
            chan2_fft = np.fft.fft(chan2)
            chan2_fft[:index-40] = -150
            chan2_fft[index+40:] = -150
            chan2_fft_shift = chan2_fft * np.exp(1.0j * dig_Beam1_phase)
            chan2 = np.fft.ifft(chan2_fft_shift, n=nSamples)
            chan2 = chan2 * win_funct
            chan2 = chan2[0:nSamples]
        
        plt.plot(chan1)
        plt.plot(chan2)
        plt.show()
        sum_chan = chan1 + chan2
        delta_chan = chan1 - chan2
        max_index = np.argmax(sum_chan)
        s_mag_sum = np.max(
            [np.abs(sum_chan[max_index]), 10 ** (-15)]
        )  # make sure this gives something >0, otherwise the log10 function will give an error
        s_mag_delta = np.max([np.abs(delta_chan[max_index]), 10 ** (-15)])
        s_dbfs_sum = 20 * np.log10(s_mag_sum / (2 ** 11))
        s_dbfs_delta = 20 * np.log10(s_mag_delta / (2 ** 11))
        total_beam_phase = total_beam_phase + (
            np.angle(sum_chan[max_index]) - np.angle(delta_chan[max_index])
        )
        total_sum = total_sum + (
            s_dbfs_sum
        )  # sum up all the loops, then we'll average
        total_delta = total_delta + (
            s_dbfs_delta
        )  # sum up all the loops, then we'll average

    PeakValue_sum = total_sum / Averages
    PeakValue_delta = total_delta / Averages
    PeakValue_beam_phase = total_beam_phase / Averages
    if np.sign(PeakValue_beam_phase) == -1:
        target_error = min(
            -0.01,
            (
                np.sign(PeakValue_beam_phase) * (PeakValue_sum - PeakValue_delta)
                + np.sign(PeakValue_beam_phase)
                * (PeakValue_sum + PeakValue_delta)
                / 2
            )
            / (PeakValue_sum + PeakValue_delta),
        )
    else:
        target_error = max(
            0.01,
            (
                np.sign(PeakValue_beam_phase) * (PeakValue_sum - PeakValue_delta)
                + np.sign(PeakValue_beam_phase)
                * (PeakValue_sum + PeakValue_delta)
                / 2
            )
            / (PeakValue_sum + PeakValue_delta),
        )
    return (
        PeakValue_sum,
        PeakValue_delta,
        PeakValue_beam_phase,
        sum_chan,
        target_error,
    )

def dbfs_cupy(raw_data):
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