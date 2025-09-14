# cal funcs
# adapted from "phaser_find_hb100.py" and "phaser_examples.py" which are found
# in: pyadi-iio > examples > phaser
# help taken from PySDR for beamsteering

import os
import pickle
import socket
import sys
import time
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
from phaser_functions import save_hb100_cal, spec_est
from scipy import signal

from adi import ad9361
from adi.cn0566 import CN0566

from phaser_functions import (
    calculate_plot,
    channel_calibration,
    gain_calibration,
    load_hb100_cal,
    phase_calibration,
)

def find_hb_100():
    # Instantiate all the Devices
    rpi_ip = "ip:phaser.local"  # IP address of the Raspberry Pi
    sdr_ip = "ip:192.168.2.1"  # "192.168.2.1, or pluto.local"  # IP address of the Transceiver Block
    my_sdr = ad9361(uri=sdr_ip)
    my_phaser = CN0566(uri=rpi_ip)

    time.sleep(0.5)

    # By default device_mode is "rx"
    my_phaser.configure(device_mode="rx")

    #  Configure SDR parameters.

    my_sdr._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "1"
    my_sdr._ctrl.debug_attrs["adi,ensm-enable-txnrx-control-enable"].value = "0"  # Disable pin control so spi can move the states
    my_sdr._ctrl.debug_attrs["initialize"].value = "1"

    my_sdr.rx_enabled_channels = [0, 1]  # enable Rx1 (voltage0) and Rx2 (voltage1)
    my_sdr._rxadc.set_kernel_buffers_count(1)  # No stale buffers to flush
    rx = my_sdr._ctrl.find_channel("voltage0")
    rx.attrs["quadrature_tracking_en"].value = "1"  # enable quadrature tracking
    my_sdr.sample_rate = int(30000000)  # Sampling rate
    my_sdr.rx_buffer_size = int(4 * 256)
    my_sdr.rx_rf_bandwidth = int(10e6)
    # We must be in manual gain control mode (otherwise we won't see the peaks and nulls!)
    my_sdr.gain_control_mode_chan0 = "manual"  # DISable AGC
    my_sdr.gain_control_mode_chan1 = "manual"
    my_sdr.rx_hardwaregain_chan0 = 0  # dB
    my_sdr.rx_hardwaregain_chan1 = 0  # dB

    my_sdr.rx_lo = int(2.0e9)  # Downconvert by 2GHz  # Receive Freq

    my_sdr.filter = "LTE20_MHz.ftr"  # Handy filter for fairly widdeband measurements

    # Make sure the Tx channels are attenuated (or off) and their freq is far away from Rx
    # this is a negative number between 0 and -88
    my_sdr.tx_hardwaregain_chan0 = int(-80)
    my_sdr.tx_hardwaregain_chan1 = int(-80)


    # Configure CN0566 parameters.
    #     ADF4159 and ADAR1000 array attributes are exposed directly, although normally
    #     accessed through other methods.


    # Set initial PLL frequency to HB100 nominal

    my_phaser.SignalFreq = 10.525e9
    my_phaser.lo = int(my_phaser.SignalFreq) + my_sdr.rx_lo


    gain_list = [64] * 8
    for i in range(0, len(gain_list)):
        my_phaser.set_chan_gain(i, gain_list[i], apply_cal=False)

    # Aim the beam at boresight (zero degrees). Place HB100 right in front of array.
    my_phaser.set_beam_phase_diff(0.0)

    # Averages decide number of time samples are taken to plot and/or calibrate system. By default it is 1.
    my_phaser.Averages = 80

    # Initialize arrays for amplitudes, frequencies
    full_ampl = np.empty(0)
    full_freqs = np.empty(0)

    # Set up range of frequencies to sweep. Sample rate is set to 30Msps,
    # for a total of 30MHz of bandwidth (quadrature sampling)
    # Filter is 20MHz LTE, so you get a bit less than 20MHz of usable
    # bandwidth. Set step size to something less than 20MHz to ensure
    # complete coverage.
    f_start = 10.0e9
    f_stop = 10.7e9
    f_step = 10e6

    for freq in range(int(f_start), int(f_stop), int(f_step)):
        #    print("frequency: ", freq)
        my_phaser.SignalFreq = freq
        my_phaser.frequency = (
            int(my_phaser.SignalFreq) + my_sdr.rx_lo
        ) // 4  # PLL feedback via /4 VCO output

        data = my_sdr.rx()
        data_sum = data[0] + data[1]
        #    max0 = np.max(abs(data[0]))
        #    max1 = np.max(abs(data[1]))
        #    print("max signals: ", max0, max1)
        ampl, freqs = spec_est(data_sum, 30000000, ref=2 ^ 12, plot=False)
        ampl = np.fft.fftshift(ampl)
        ampl = np.flip(ampl)  # Just an experiment...
        freqs = np.fft.fftshift(freqs)
        freqs += freq
        full_freqs = np.concatenate((full_freqs, freqs))
        full_ampl = np.concatenate((full_ampl, ampl))
        sleep(0.1)
    full_freqs /= 1e9  # Hz -> GHz

    peak_index = np.argmax(full_ampl)
    peak_freq = full_freqs[peak_index]
    print("Peak frequency found at ", full_freqs[peak_index], " GHz.")

    plt.figure(2)
    plt.title("Full Spectrum, peak at " + str(full_freqs[peak_index]) + " GHz.")
    plt.plot(full_freqs, full_ampl, linestyle="", marker="o", ms=2)
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Signal Strength")
    plt.show()
    print("You may need to close plot to continue...")

    prompt = input("Save cal file? (y or n)")
    if prompt.upper() == "Y":
        save_hb100_cal(peak_freq * 1e9)

    del my_sdr
    del my_phaser
    return True

def full_cal():
    try:
        import config_custom as config  # this has all the key parameters that the user would want to change (i.e. calibration phase and antenna element spacing)

        print("Found custom config file")
    except:
        print("Didn't find custom config, looking for default.")
        try:
            import config as config
        except:
            print("Make sure config.py is in this directory")
            sys.exit(0)

    colors = ["black", "gray", "red", "orange", "yellow", "green", "blue", "purple"]


    def do_cal_channel(my_phaser):
        my_phaser.set_beam_phase_diff(0.0)
        channel_calibration(my_phaser, verbose=True)


    def do_cal_gain(my_phaser):
        my_phaser.set_beam_phase_diff(0.0)
        #    plot_data = my_phaser.gain_calibration(verbose=True)  # Start Gain Calibration
        plot_data = gain_calibration(my_phaser, verbose=True)  # Start Gain Calibration
        plt.figure(4)
        plt.title("Gain calibration FFTs")
        plt.xlabel("FFT Bin number")
        plt.ylabel("Amplitude (ADC counts)")
        for i in range(0, 8):
            plt.plot(plot_data[i], color=colors[i])
        plt.show()


    def do_cal_phase(my_phaser):
        # PhaseValues, plot_data = my_phaser.phase_calibration(
        #     verbose=True
        # )  # Start Phase Calibration
        PhaseValues, plot_data = phase_calibration(
            my_phaser, verbose=True
        )  # Start Phase Calibration
        plt.figure(5)
        plt.title("Phase sweeps of adjacent elements")
        plt.xlabel("Phase difference (degrees)")
        plt.ylabel("Amplitude (ADC counts)")
        for i in range(0, 7):
            plt.plot(PhaseValues, plot_data[i], color=colors[i])
        plt.show()


    
    # Instantiate all the Devices
    rpi_ip = "ip:phaser.local"  # IP address of the Raspberry Pi
    sdr_ip = "ip:192.168.2.1"  # "192.168.2.1, or pluto.local"  # IP address of the Transceiver Block
    my_sdr = ad9361(uri=sdr_ip)
    my_phaser = CN0566(uri=rpi_ip)

    my_phaser.sdr = my_sdr  # Set my_phaser.sdr

    time.sleep(0.5)


    # By default device_mode is "rx"
    my_phaser.configure(device_mode="rx")


    my_phaser.SDR_init(30000000, config.Tx_freq, config.Rx_freq, 3, -6, 1024)

    my_phaser.load_channel_cal()
    # First crack at compensating for channel gain mismatch
    my_phaser.sdr.rx_hardwaregain_chan0 = (
        my_phaser.sdr.rx_hardwaregain_chan0 + my_phaser.ccal[0]
    )
    my_phaser.sdr.rx_hardwaregain_chan1 = (
        my_phaser.sdr.rx_hardwaregain_chan1 + my_phaser.ccal[1]
    )

    # Set up receive frequency. When using HB100, you need to know its frequency
    # fairly accurately. Use the cn0566_find_hb100.py script to measure its frequency
    # and write out to the cal file. IF using the onboard TX generator, delete
    # the cal file and set frequency via config.py or config_custom.py.

    try:
        my_phaser.SignalFreq = load_hb100_cal()
        print("Found signal freq file, ", my_phaser.SignalFreq)
    except:
        my_phaser.SignalFreq = config.SignalFreq
        print("No signal freq found, keeping at ", my_phaser.SignalFreq)
        print("And using TX path. Make sure antenna is connected.")
        config.use_tx = True  # Assume no HB100, use TX path.
        
    """my_phaser.SignalFreq = config.SignalFreq
    print("No signal freq found, keeping at ", my_phaser.SignalFreq)
    print("And using TX path. Make sure antenna is connected.")
    config.use_tx = True  # Assume no HB100, use TX path."""
    #  Configure SDR parameters.

    my_sdr.filter = "LTE20_MHz.ftr"  # Load LTE 20 MHz filter


    # use_tx = config.use_tx
    choice = input("use transmit circuitry or not?? [y,n]")
    if choice=='y':
        use_tx=True
    else:
        use_tx=False

    if use_tx is True:
        # To use tx path, set chan1 gain "high" keep chan0 attenuated.
        my_sdr.tx_hardwaregain_chan0 = int(
            -88
        )  # this is a negative number between 0 and -88
        my_sdr.tx_hardwaregain_chan1 = int(-3)
        my_sdr.tx_lo = config.Tx_freq  # int(2.2e9)

        my_sdr.dds_single_tone(
            int(2e6), 0.9, 1
        )  # sdr.dds_single_tone(tone_freq_hz, tone_scale_0to1, tx_channel)
    else:
        # To disable rx, set attenuation to a high value and set frequency far from rx.
        my_sdr.tx_hardwaregain_chan0 = int(
            -88
        )  # this is a negative number between 0 and -88
        my_sdr.tx_hardwaregain_chan1 = int(-88)
        my_sdr.tx_lo = int(1.0e9)

    # To use tx path, set chan1 gain "high" keep chan0 attenuated.
    my_sdr.tx_hardwaregain_chan0 = int(
        -88
    )  # this is a negative number between 0 and -88
    my_sdr.tx_hardwaregain_chan1 = int(-88)
    my_sdr.tx_lo = config.Tx_freq  # int(2.2e9)

    my_sdr.dds_single_tone(
        int(2e6), 0.9, 1
    )  # sdr.dds_single_tone(tone_freq_hz, tone_scale_0to1, tx_channel)

    # Configure CN0566 parameters.
    #     ADF4159 and ADAR1000 array attributes are exposed directly, although normally
    #     accessed through other methods.


    my_phaser.frequency = (10492000000 + 2000000000) // 4 #6247500000//2

    # Onboard source w/ external Vivaldi
    my_phaser.frequency = (
        int(my_phaser.SignalFreq) + config.Rx_freq
    ) // 4  # PLL feedback via /4 VCO output
    my_phaser.freq_dev_step = 5690
    my_phaser.freq_dev_range = 0
    my_phaser.freq_dev_time = 0
    my_phaser.powerdown = 0
    my_phaser.ramp_mode = "disabled"

    #  If you want to use previously calibrated values load_gain and load_phase values by passing path of previously
    #  stored values. If this is not done system will be working as uncalibrated system.
    #  These will fail gracefully and default to no calibration if files not present.

    my_phaser.load_gain_cal("gain_cal_val.pkl")
    my_phaser.load_phase_cal("phase_cal_val.pkl")

    # This can be useful in Array size vs beam width experiment or beamtappering experiment.
    #     Set the gain of outer channels to 0 and beam width will increase and so on.

    # To set gain of all channels with different values.
    #     Here's where you would apply a window / taper function,
    #     but we're starting with rectangular / SINC1.

    gain_list = [127, 127, 127, 127, 127, 127, 127, 127]
    for i in range(0, len(gain_list)):
        my_phaser.set_chan_gain(i, gain_list[i], apply_cal=True)

    # Averages decide number of time samples are taken to plot and/or calibrate system. By default it is 1.
    my_phaser.Averages = 4

    # Aim the beam at boresight by default
    my_phaser.set_beam_phase_diff(0.0)

    
    input(
            "Calibrating gain and phase - place antenna at mechanical boresight in front of the array, then press enter..."
        )
    print("Calibrating gain mismatch between SDR channels, then saving cal file...")
    do_cal_channel(my_phaser)
    my_phaser.save_channel_cal()
    print("Calibrating Gain, verbosely, then saving cal file...")
    do_cal_gain(my_phaser)  # Start Gain Calibration
    my_phaser.save_gain_cal()  # Default filename
    print("Calibrating Phase, verbosely, then saving cal file...")
    do_cal_phase(my_phaser)  # Start Phase Calibration
    my_phaser.save_phase_cal()  # Default filename
    print("Done calibration")
        
    del my_sdr
    del my_phaser
    return True

def phaser_beamsteer():
    
    phase_cal = pickle.load(open("phase_cal_val.pkl", "rb"))
    gain_cal = pickle.load(open("gain_cal_val.pkl", "rb"))
    signal_freq = pickle.load(open("hb100_freq_val.pkl", "rb"))
    d = 0.014  # element to element spacing of the antenna

    phaser = CN0566(uri="ip:phaser.local")
    sdr = ad9361(uri="ip:192.168.2.1")
    phaser.sdr = sdr
    print("PlutoSDR and CN0566 connected!")

    time.sleep(0.5) # recommended by Analog Devices

    phaser.configure(device_mode="rx")

    # Set all antenna elements to half scale - a typical HB100 will have plenty of signal power.
    
    #gain_list = [53, 73, 125, 119, 71, 114, 68, 43]
    #gain_list = [127]*8
    #gain_list = [42,    80,   115,    81,    69,    39,    29,    29]
    #gain_list = [29,    83,   120,    79,   113,    86,    32,    36]
    #gain_list  = [35,   102,   108,    90,   121,    57,   112,    28]
    #gain_list = [36,    50,    71,    71,   106,    69,    70,    55]
    
    #gain_list = [25,    29,    86,   124,   118,    66,   113,    69]
    #gain_list = [25,    98,    86,   124,   118,    66,   113,    58]
    #gain_list = [21,    26,    74,    97,   102,   112,   101,    34]
    #gain_list = [93,   120,    65,   115,    80,    70,    66,    51]
    
    #gain_list = [50,    78,   82,   116,   101,    85,    53,    47]
    #gain_list = [40,    56,    83,   101,    49,    28,    34,    35]
    
    
    #gain_list = [50, 78, 82, 116, 101, 85, 53, 47]
    gain_list = [25, 98, 86, 124, 118, 66, 113, 58]
    for i in range(0,len(gain_list)):
        phaser.set_chan_gain(i, gain_list[i], apply_cal=True)

    # Aim the beam at boresight (zero degrees)
    phaser.set_beam_phase_diff(0.0)

    # Misc SDR settings, not super critical to understand
    sdr._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "1"
    sdr._ctrl.debug_attrs["adi,ensm-enable-txnrx-control-enable"].value = "0" # Disable pin control so spi can move the states
    sdr._ctrl.debug_attrs["initialize"].value = "1"
    sdr.rx_enabled_channels = [0, 1] # enable Rx1 and Rx2
    sdr._rxadc.set_kernel_buffers_count(1) # No stale buffers to flush
    sdr.tx_hardwaregain_chan0 = int(-80) # Make sure the Tx channels are attenuated (or off)
    sdr.tx_hardwaregain_chan1 = int(-80)

    # These settings are basic PlutoSDR settings we have seen before
    sample_rate = 30e6
    sdr.sample_rate = int(sample_rate)
    sdr.rx_buffer_size = int(1024)  # samples per buffer
    sdr.rx_rf_bandwidth = int(10e6)  # analog filter bandwidth

    # Manually gain (no automatic gain control) so that we can sweep angle and see peaks/nulls
    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"
    sdr.rx_hardwaregain_chan0 = 5 # dB, 0 is the lowest gain.  the HB100 is pretty loud
    sdr.rx_hardwaregain_chan1 = 5 # dB

    sdr.rx_lo = int(2.2e9) # The Pluto will tune to this freq

    # Set the Phaser's PLL (the ADF4159 onboard) to downconvert the HB100 to 2.2 GHz plus a small offset
    offset = 1000000 # add a small arbitrary offset just so we're not right at 0 Hz where there's a DC spike
    phaser.lo = int(signal_freq + sdr.rx_lo - offset)
    powers = [] # main DOA result
    angle_of_arrivals = []
    for phase in np.arange(-180, 181, 1): # sweep over angle
        phase_shift = [0]*8
        # set phase difference between the adjacent channels of devices
        for i in range(8):
            
            channel_phase = (phase * i + phase_cal[i]) % 360.0 # Analog Devices had this forced to be a multiple of phase_step_size (2.8125 or 360/2**6bits) but it doesn't seem nessesary
            phaser.elements.get(i + 1).rx_phase = channel_phase
            phase_shift[i] = float(channel_phase)
            
        print(phase_shift)
        phaser.latch_rx_settings() # apply settings
        steer_angle = np.degrees(np.arcsin(max(min(1, (3e8 * np.radians(phase)) / (2 * np.pi * signal_freq * phaser.element_spacing)), -1))) # arcsin argument must be between 1 and -1, or numpy will throw a warning
        # If you're looking at the array side of Phaser (32 squares) then add a *-1 to steer_angle
        angle_of_arrivals.append(steer_angle)
        data = phaser.sdr.rx() # receive a batch of samples
            
        data_sum = data[0] + data[1] # sum the two subarrays (within each subarray the 4 channels have already been summed)
        #data_sum = data[0] + data[1]*(np.conjugate(digi_shift)) # sum the two subarrays (within each subarray the 4 channels have already been summed)
        power_dB = 10*np.log10(np.sum(np.abs(data_sum)**2))
        powers.append(power_dB)
        # in addition to just taking the power in the signal, we could also do the FFT then grab the value of the max bin, effectively filtering out noise, results came out almost exactly the same in my tests
        #PSD = 10*np.log10(np.abs(np.fft.fft(data_sum * np.blackman(len(data_sum))))**2) # in dB
        
    
    powers -= np.max(powers) # normalize so max is at 0 dB
    
    plt.plot(angle_of_arrivals, powers, '.-')
    plt.xlabel("Angle of Arrival")
    plt.ylabel("Magnitude [dB]")
    plt.show()
    
    del sdr
    del phaser
    return powers,angle_of_arrivals,data_sum
#%%
find_hb_100()
#%%
full_cal()
#%%
data,angles,signal = phaser_beamsteer()