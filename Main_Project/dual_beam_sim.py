
import os
import pickle
import socket
import sys
import time
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from adi import ad9361
from adi.cn0566 import CN0566

from custom_classes import PhasedArrayDualBeam, phased_array_dual_beam_weights

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
phaser.load_gain_cal()
phaser.load_phase_cal()


gain_list = [127]*8

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
sdr.rx_hardwaregain_chan0 = 20 # dB, 0 is the lowest gain.  the HB100 is pretty loud
sdr.rx_hardwaregain_chan1 = 20 # dB

sdr.rx_lo = int(2.2e9) # The Pluto will tune to this freq

# Set the Phaser's PLL (the ADF4159 onboard) to downconvert the HB100 to 2.2 GHz plus a small offset
offset = 1000000 # add a small arbitrary offset just so we're not right at 0 Hz where there's a DC spike
phaser.lo = int(signal_freq + sdr.rx_lo - offset)

# weights for beams at -27 and 12 degrees
#weights_mag = np.array([0.19448888, 0.18955276, 0.00248433, 0.19206923, 0.19206923, 0.00248433, 0.18955276, 0.19448888])
#weights_phase = np.array([ -102.48517519,  -124.63226799,  +3.2206392,   +11.0735464,    -11.0735464,   -33.2206392,  +124.63226799, +102.48517519])

def form_dual_beams(angle_list, my_phaser, sig_freq):
    weights_raw = phased_array_dual_beam_weights(angle_list, sig_freq, 8, None, method='mvdr')
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

angle_list = [-30, 20]
(weights_raw, weights_mag, weights_degrees, phaser) = form_dual_beams(angle_list, phaser, signal_freq)

powers = []
    
#%% Take Data
# Record Data After rotating the phaser kit, by specific degrees, on a gimble 
data = sdr.rx()
data = data[0] + data[1]
power_dB = 10*np.log10(np.sum(np.abs(data)**2))
powers.append(power_dB)
print(power_dB)


