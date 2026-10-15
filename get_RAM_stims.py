import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def get_RAM_stims(fs, fRAM, plot=False):
    """
    Generating stimulation waveform for RAM (Rectangular Amplitude Modulation)
    Based on Sarineh Keshishzadeh 10.11.2021.

    Parameters:
    -----------
    fs : float
        Sampling frequency [Hz]
    fRAM : array_like
        [Mx1] vector of RAM carrier frequencies [Hz]
        M - number of stimuli to generate
    plot : bool, optional
        If True, plots the reference SAM tone and the generated RAM
        stimulus (envelope + waveform) for the first carrier frequency
        in fRAM. Default False.

    Returns:
    --------
    stim : numpy.ndarray
        [MxN] matrix of the generated stimuli
        N - number of samples in the stimuli

    Created by:
    -----------
    Brent Nissens
    October 22, 2025
    """
    
    # Reference SAM parameters (RAM stimuli are calibrated with reference of SAM tone)
    level = 70                    # Stimulation intensity [dBSPL]
    p0 = 20e-6                    # Reference sound pressure level [Pa]
    fSAM = 4000                   # SAM carrier frequency [Hz]
    
    # RAM parameters
    fMod = 110                    # modulation frequency [Hz]
    md = 1                        # modulation depth
    phi = 3 * np.pi / 2           # Carrier phase in [rad]
    dur = 0.4                     # Stimulus duration in [s]
    
    # Get the time vector
    n = round(dur * fs)           # number of samples
    tVec = np.arange(n) / fs      # time vector
    
    # Generate the reference SAM tone
    carrierSAM = np.sin(2 * np.pi * fSAM * tVec)            ###Pure tone carrier signal before any modulation
    modSAM = np.sin(2 * np.pi * fMod * tVec + phi)          ###this is the modulator
    SAM = carrierSAM * (1 + md * modSAM)                    ###
    
    
    # Calculate RMS for normalization
    SAM_rms = np.sqrt(np.mean(SAM**2))
    SAM = p0 * 10**(level/20) * SAM / SAM_rms  # adjust level such that 1Pa is the full-scale signal
    print(SAM)
    
    
    # RAM stimuli, for modulator 110Hz and carrier frequencies in fRAM
    # Generate square wave modulator (25% duty cycle) - matches MATLAB square(x, 25)
    modRAM = signal.square(2 * np.pi * fMod * tVec + phi, duty=0.25)
    
    # Initialize output matrix
    stim = np.zeros((len(fRAM), n))
    
    # Generate RAM stimuli for each carrier frequency
    RAM_raw_plot = None
    for i in range(len(fRAM)):
        carrierRAM = np.sin(2 * np.pi * fRAM[i] * tVec)
        RAM = carrierRAM * (1 + md * modRAM)
        if i == 0:
            RAM_raw_plot = RAM  # keep the unscaled RAM for the first carrier, for plotting

        # Scale the RAM signal such that it has the same max value as the SAM
        stim[i, :] = np.max(SAM) / np.max(RAM) * RAM

    if plot:
        t_end = 3 / fMod
        mask = tVec <= t_end
        t_ms = tVec[mask] * 1000
        print(mask)

        _, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

        axes[0].plot(t_ms, carrierSAM[mask], linewidth=0.8)
        axes[0].set_title(f'carrierSAM(t) = sin(2pi x {fSAM} x t)')
        axes[0].set_ylabel('amplitude')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t_ms, modSAM[mask], linewidth=0.8)
        axes[1].set_title(f'modSAM(t) = sin(2pi x {fMod} x t + phi)')
        axes[1].set_ylabel('amplitude')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t_ms, SAM[mask], linewidth=0.8)
        axes[2].set_title(f'SAM(t) = carrierSAM x (1 + md x modSAM)')
        axes[2].set_ylabel('Pa')
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(t_ms, carrierRAM[mask], color = 'red', linewidth=0.8)
        axes[3].set_title(f'RAM(t) = carrierRAM x (1 + md x modRAM)')
        axes[3].set_ylabel('amplitude')
        axes[3].grid(True, alpha=0.3)

        axes[4].plot(t_ms, stim[0, mask], color = 'red', linewidth=0.8)
        axes[4].set_title(f'stim(t) = max(SAM)/max(RAM) x RAM')
        axes[4].set_ylabel('Pa')
        axes[4].set_xlabel('time [ms]')
        axes[4].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return stim


def get_click_stim(fs, n_stim=1, level=70, click_dur=100e-6, onset=0.005, dur=0.025, rate=None, plot=False):
    
    p0 = 20e-6                                # Reference sound pressure level [Pa]
    amp = p0 * 10**(level/20) * np.sqrt(2)    # peak amplitude [Pa]

    n = round(dur * fs)
    tVec = np.arange(n) / fs
    nclick = max(1, round(click_dur * fs))

    if rate is None:
        onsets = [round(onset * fs)]
    else:
        onsets = np.arange(round(onset * fs), n, round(fs / rate))

    stim = np.zeros((n_stim, n))
    for i0 in onsets:
        stim[:, i0:i0 + nclick] = amp

    if plot:
        plt.plot(tVec * 1000, stim[0])
        plt.xlabel('time [ms]')
        plt.ylabel('Pa')
        plt.title('Click stimulus')
        plt.grid(True, alpha=0.3)
        plt.show()

    return stim


if __name__ == "__main__":
    fs = 44100
    fRAM = np.array([4000])
    stim = get_RAM_stims(fs, fRAM, plot=True)
    print(f"Generated {stim.shape[0]} RAM stimuli with {stim.shape[1]} samples each")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Carrier frequencies: {fRAM} Hz")