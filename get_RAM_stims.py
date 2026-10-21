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


def get_pip_chirp_stim(fs, kind='pip', n_stim=1, level=70, freq=4000, f_sweep=(100, 8000),          ##Tone pip: very short bust of sound at one fixed frequency
                       method='linear', fMod=0, ramp_dur=1e-3, onset=0.005, dur=0.025,       ##
                       plot=False):
    """
    Tone pip ('pip') or chirp ('chirp') stimulus, same level convention as get_click_stim
    (peak amplitude = peak of a sine with rms level `level` dBSPL).

    freq    : pip frequency [Hz]
    f_sweep : (f_start, f_end) of the chirp [Hz]
    method  : chirp sweep, 'linear' or 'logarithmic' (scipy.signal.chirp)
    fMod    : modulation (envelope) frequency [Hz]: the pip / chirp lasts one period of it, 1/fMod
              (200 Hz = 5 ms), so a higher fMod gives a shorter stimulus
    ramp_dur: cos^2 on/off ramp [s] (at most half the stimulus)
    """

    p0 = 20e-6                                # Reference sound pressure level [Pa]
    amp = p0 * 10**(level/20) * np.sqrt(2)    # peak amplitude [Pa]

    n = round(dur * fs)
    tVec = np.arange(n) / fs
    i0 = round(onset * fs)
    nstim = round(fs / fMod)                  # one period of the modulation
    nramp = min(round(ramp_dur * fs), nstim // 2)
    if i0 + nstim > n:
        raise ValueError(f"fMod = {fMod} Hz gives a {1000 / fMod:.1f} ms stimulus, onset + that exceeds dur "
                         f"({dur * 1000:.0f} ms): raise fMod or dur")

    t = np.arange(nstim) / fs
    if kind == 'pip':
        wave = np.sin(2 * np.pi * freq * t)
    elif kind == 'chirp':
        wave = signal.chirp(t, f0=f_sweep[0], t1=t[-1], f1=f_sweep[1], method=method, phi=-90)
    else:
        raise ValueError("kind must be 'pip' or 'chirp'")

    # cos^2 onset/offset ramps
    env = np.ones(nstim)
    ramp = np.sin(np.linspace(0, np.pi / 2, nramp))**2
    env[:nramp] = ramp
    env[nstim - nramp:] = ramp[::-1]

    stim = np.zeros((n_stim, n))
    stim[:, i0:i0 + nstim] = amp * env * wave

    if plot:
        _, axes = plt.subplots(2, 1, figsize=(10, 7))

        axes[0].plot(tVec * 1000, stim[0])
        axes[0].set_xlabel('time [ms]')
        axes[0].set_ylabel('Pa')
        axes[0].set_title('Tone pip' if kind == 'pip' else 'Chirp')
        axes[0].grid(True, alpha=0.3)

        nfft = 2**14
        spec = np.abs(np.fft.rfft(stim[0], nfft))
        axes[1].plot(np.fft.rfftfreq(nfft, 1 / fs), 20 * np.log10(spec / spec.max() + 1e-12))
        axes[1].set_xlim(0, 12000)
        axes[1].set_ylim(-80, 5)
        axes[1].set_xlabel('frequency [Hz]')
        axes[1].set_ylabel('magnitude [dB re max]')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return stim


if __name__ == "__main__":
    fs = 44100
    fRAM = np.array([4000])
    stim = get_RAM_stims(fs, fRAM, plot=True)
    stim2 = get_click_stim(fs, plot=True)
    stim3 = get_pip_chirp_stim(fs, plot=True)
    stim4 = get_pip_chirp_stim(fs, kind = 'chirp', plot=True)
    print(f"Generated {stim.shape[0]} RAM stimuli with {stim.shape[1]} samples each")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Carrier frequencies: {fRAM} Hz")