#!/usr/bin/env python3
"""
Easy Model Runner with EFR Calculation

This script provides a simple interface to run the model2018 and calculate EFR (Envelope Following Response).

Usage:
    python easy_model_run.py

Features:
- Generates RAM stimulus automatically
- Loads poles profile (configurable)
- Runs model2018 simulation
- Calculates EFR using FFT analysis
- Displays results clearly
- Saves output to files

Created by: Brent Nissens
Based on: Verhulst et al. 2018 model and FullSimulationRAM.py
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import io
import time
import contextlib
import matplotlib.colors as mcolors
from get_RAM_stims import get_RAM_stims, get_click_stim
from model2018 import model2018

from fractions import Fraction
from scipy import signal as sps

def calculate_EFR(output):
    """
    Calculate the EFR from model output using FFT analysis.
    
    This function computes the EFR (Envelope Following Response) harmonics sum 
    from simulation output, similar to the calculation in FullSimulationRAM.py.
    
    Parameters:
    -----------
    output : ModelOutput
        Output from model2018 function
        
    Returns:
    --------
    float
        EFR value in microvolts (μV)
    """
    try:
        # Handle the case where output is a list containing a dictionary
        if isinstance(output, list) and len(output) > 0:
            output = output[0]  # Get the first (and likely only) element
        
        # Extract relevant waveforms and sampling frequency from the output object/struct.
        fs = float(output.fs_an)
        w1 = output.w1.flatten()
        w3 = output.w3.flatten()
        w5 = output.w5.flatten()

        # Combine waves to get EFR
        EFR = w1 + w3 + w5

        # Fourier Transform
        L = len(EFR)
        Y = np.fft.fft(EFR)
        P2 = np.abs(Y / L)
        P1 = P2[:L//2 + 1]
        P1[1:-1] = 2 * P1[1:-1]
        f = fs * np.arange(L//2 + 1) / L

        # Find indices for 4 harmonics of 110 Hz (fundamental modulation frequency)
        fundamental = 110  # Hz
        num_harmonics = 4
        harmonics = np.arange(1, num_harmonics + 1) * fundamental
        idx = []
        for harmonic in harmonics:
            idx.append(np.argmin(np.abs(f - harmonic)))
        
        # Calculate sum of harmonics and convert to microV
        harmonic_sum = np.sum(P1[idx]) * 1e6

        return harmonic_sum, f, P1, harmonics, idx

    except Exception as e:
        print(f"Error calculating EFR: {e}")
        return np.nan, None, None, None, None

def run_easy_model(carrier_freq=4000, poles_profile='Flat00', show_plots=True, save_results=True, stim='click'):
    """
    Easy interface to run the model2018 and calculate EFR.
    
    Parameters:
    -----------
    carrier_freq : float, optional
        Carrier frequency for RAM stimulus in Hz (default: 4000)
    poles_profile : str, optional
        Name of poles profile folder in ./Poles/ directory (default: 'Flat00')
    show_plots : bool, optional
        Whether to show plots of results (default: True)
    save_results : bool, optional
        Whether to save results to MAT files (default: True)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'efr_value': EFR value in μV
        - 'output': Full model output
        - 'stimulus': Generated stimulus
        - 'frequency_spectrum': Frequency array for FFT
        - 'power_spectrum': Power spectrum
    """
    
    print("="*60)
    print("Easy Model2018 Runner with EFR Calculation")
    print("="*60)
    
    # Configuration
    fs = 1e5  # Sampling frequency
    fRAM = np.array([carrier_freq])
    
    print(f"Carrier frequency: {carrier_freq} Hz")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Poles profile: {poles_profile}")
    print()
    
    # Generate stimulus
    print("Generating RAM stimulus...")
    try:
        if stim == 'click':
            stimulus = get_click_stim(fs, level=80, onset=0.03, dur=0.065)
        else:
            stimulus = get_RAM_stims(fs, fRAM)

        print(f"✓ Generated stimulus: {stimulus.shape[1]} samples, duration: {stimulus.shape[1]/fs:.3f} s")
    except Exception as e:
        print(f"✗ Error generating stimulus: {e}")
        return None
    
    # Load poles profile
    poles_path = f'./Poles/{poles_profile}/StartingPoles.dat'
    print(f"Loading poles from: {poles_path}")
    try:
        sheraP = np.loadtxt(poles_path)
        # Take only the first row if there are multiple rows
        if sheraP.ndim > 1:
            sheraP = sheraP[0, :]
        print(f"✓ Loaded poles profile: {len(sheraP)} values")
    except Exception as e:
        print(f"✗ Error loading poles: {e}")
        print(f"Available poles profiles in ./Poles/:")
        try:
            poles_dirs = [d for d in os.listdir('./Poles/') if os.path.isdir(f'./Poles/{d}')]
            for d in sorted(poles_dirs):
                print(f"  - {d}")
        except:
            print("  Could not list poles directories")
        return None
    
    # Run model
    print("\nRunning model2018...")
    try:
        results = model2018(
            stimulus, 
            fs, 
            fc='abr',                    # Use ABR frequency points
            irregularities=1,            # Enable irregularities
            storeflag='evihmlbw',        # Store all relevant outputs
            subject=1,                   # Subject seed
            sheraPo=sheraP,              # Poles profile
            IrrPct=0.05,                 # 5% irregularities
            non_linear_type='vel',       # Velocity-based nonlinearity
            nH=13,                       # High spontaneous rate fibers
            nM=3,                        # Medium spontaneous rate fibers
            nL=3,                        # Low spontaneous rate fibers
            clean=1,                     # Clean parameter
            data_folder='./'             # Data folder
        )
        print("✓ Model simulation completed successfully!")
    except Exception as e:
        print(f"✗ Error running model: {e}")
        return None
    
    # Calculate EFR
    print("\nCalculating EFR...")
    try:
        efr_value, f, P1, harmonics, harmonic_idx = calculate_EFR(results)
        if not np.isnan(efr_value):
            print(f"✓ EFR calculated: {efr_value:.4f} μV")
        else:
            print("✗ EFR calculation failed")
            return None
    except Exception as e:
        print(f"✗ Error calculating EFR: {e}")
        return None
    
    # Display harmonic details
    print("\nHarmonic Analysis:")
    for i, (harm_freq, idx) in enumerate(zip(harmonics, harmonic_idx)):
        power_uv = P1[idx] * 1e6
        print(f"  {harm_freq} Hz (harmonic {i+1}): {power_uv:.4f} μV")
    
    # Save results if requested
    if save_results:
        print("\nSaving results...")
        try:
            output = results[0]
            
            # Save EFR and waves
            EFR_combined = output.w1 + output.w3 + output.w5
            sio.savemat('easy_model_EFR.mat', {
                'EFR': EFR_combined,
                'w1': output.w1,
                'w3': output.w3,
                'w5': output.w5,
                'efr_value_uV': efr_value,
                'fs': output.fs_an,
                'carrier_freq': carrier_freq,
                'poles_profile': poles_profile
            })
            
            # Save velocity and other outputs
            sio.savemat('easy_model_output.mat', {
                'v': output.v,
                'cf': output.cf,
                'stimulus': stimulus,
                'fs_bm': output.fs_bm
            })
            
            print("✓ Results saved to:")
            print("  - easy_model_EFR.mat (EFR and waves)")
            print("  - easy_model_output.mat (full model output)")
            
        except Exception as e:
            print(f"✗ Error saving results: {e}")
    
    # Create plots if requested
    if show_plots and f is not None:
        print("\nGenerating plots...")
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'Model2018 Results - Carrier: {carrier_freq} Hz, Profile: {poles_profile}', fontsize=14)
            
            # Plot 1: Stimulus
            t_stim = np.arange(len(stimulus[0])) / fs
            axes[0,0].plot(t_stim[:1000], stimulus[0][:1000])  # Show first 1000 samples
            axes[0,0].set_title('RAM Stimulus (first 10 ms)')
            axes[0,0].set_xlabel('Time (s)')
            axes[0,0].set_ylabel('Amplitude')
            axes[0,0].grid(True)
            
            # Plot 2: EFR waveform
            output = results[0]
            EFR_combined = output.w1 + output.w3 + output.w5
            t_efr = np.arange(len(EFR_combined)) / output.fs_an
            axes[0,1].plot(t_efr, EFR_combined)
            axes[0,1].set_title('EFR Waveform (w1+w3+w5)')
            axes[0,1].set_xlabel('Time (s)')
            axes[0,1].set_ylabel('Amplitude')
            axes[0,1].grid(True)
            
            # Plot 3: Frequency spectrum
            axes[1,0].semilogy(f[:2000], P1[:2000])  # Show up to 400 Hz
            for i, idx in enumerate(harmonic_idx):
                axes[1,0].semilogy(f[idx], P1[idx], 'ro', markersize=8, 
                                 label=f'{harmonics[i]} Hz')
            axes[1,0].set_title('FFT Spectrum with Harmonics')
            axes[1,0].set_xlabel('Frequency (Hz)')
            axes[1,0].set_ylabel('Power')
            axes[1,0].grid(True)
            axes[1,0].legend()
            axes[1,0].set_xlim(0, 500)
            
            # Plot 4: Individual waves
            axes[1,1].plot(t_efr, output.w1, label='Wave 1', alpha=0.7)
            axes[1,1].plot(t_efr, output.w3, label='Wave 3', alpha=0.7)
            axes[1,1].plot(t_efr, output.w5, label='Wave 5', alpha=0.7)
            axes[1,1].set_title('Individual ABR Waves')
            axes[1,1].set_xlabel('Time (s)')
            axes[1,1].set_ylabel('Amplitude')
            axes[1,1].legend()
            axes[1,1].grid(True)
            
            plt.tight_layout()
            plt.savefig('easy_model_results.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            print("✓ Plots saved as 'easy_model_results.png'")
            
        except Exception as e:
            print(f"✗ Error creating plots: {e}")
    
    # Prepare return dictionary
    result_dict = {
        'efr_value': efr_value,
        'output': results[0],
        'stimulus': stimulus,
        'frequency_spectrum': f,
        'power_spectrum': P1,
        'harmonics': harmonics,
        'carrier_freq': carrier_freq,
        'poles_profile': poles_profile
    }
    
    print(f"\n" + "="*60)
    print(f"SUMMARY: EFR = {efr_value:.4f} μV")
    print("="*60)
    
    return result_dict

def abr_convolution_split(output, click_onset=0.03, fMod=110, window=None, dur=None,
                          latencies=(0.0, 0.0e-3, 0.0e-3, 0.0e-3), show_plots=True, single_click=False):
    """
    Same as abr_convolution_click, but the inhibition of the CN and IC is split off (by the model itself,
    ic_cn2018.splitNuclei) and timed separately:
    Wave_func(t) = w1(t - Lat_1) + w3_pos(t - Lat_2) + w5_pos(t - Lat_3) + w3_neg(t - Lat_4) + w5_neg(t - Lat_4)
    Each term is the delta train shifted by its latency, convolved with the response of that component
    to ONE click (baseline removed). The inhibitory parts have no built-in delay (the model's 10 ms CN and
    0.2 ms IC inhibition delays are removed, see ic_cn2018.splitNuclei), so Lat_4 is when the inhibition starts
    after the click.

    Parameters:
    -----------
    output : ModelOutput or list containing it
        model2018 output for ONE click (needs w1, w3_pos, w3_neg, w5_pos, w5_neg and fs_an, i.e. 'w' in storeflag)
    click_onset : float, optional
        Time of the click in the stimulus [s]; must equal 'onset' of get_click_stim (default 0.03). Keep it
        above 25 ms, the inhibitory filters need that long to settle before the click
    fMod : float, optional
        Rate of the delta train = click rate of the simulated train [Hz] (default 110)
    window : float or None, optional
        Length of the unit responses after the click [s]. None = until 1 ms before the end of the model
        output (default None)
    dur : float or None, optional
        Duration of the delta train and of the convolved output [s]. None = one period (1/fMod): a single
        click, so each component is generated once and the output stops after one period (default None)
    latencies : tuple of 4 floats, optional
        (Lat_1, Lat_2, Lat_3, Lat_4) [s]: delay of wave I, excitatory III, excitatory V and of both
        inhibitory parts (III and V together). Wave I and the excitatory parts keep their natural timing and
        the latency is added to it, the inhibitory parts start at Lat_4 after the click
        (default (0, 1.7e-3, 4.3e-3, 7e-3))
    show_plots : bool, optional
        Plot the results and save them as 'abr_convolution_split.png' (default True)
    single_click : bool, optional
        True = the delta train holds only the first click, whatever fMod is, so every response runs on for
        the whole of dur instead of being overlapped by the next click (default False)

    Returns:
    --------
    dict with 'unit_responses', 'rate_functions', 'convolved' (one array per component 'I', 'III+', 'V+',
    'III-', 'V-'), 'delta_train' and 'total' (= Wave_func, 2xN arrays, row 0 = time [s]), 'latencies' and
    'fs' (upsampled sampling frequency [Hz])
    """

    if isinstance(output, (list, tuple)):
        output = output[0]
    if click_onset < 0.025:
        print("Warning: click_onset < 25 ms, the inhibitory filters have not settled before the click, "
              "so the baseline of the inhibitory parts is off")
    fs_in = float(output.fs_an)

    up = Fraction(int(round(fs_in)), int(round(fMod))).denominator     # 11 for 20 kHz and 110 Hz
    fs = fs_in * up
    n_edge = int(round(1e-3 * fs_in))                   # 1 ms: baseline window before the click, guard at the end
    i_click_in = int(round(click_onset * fs_in))        # sample of the click in the model output

    # components with the pre-click baseline removed (before the upsampling, so the filter does not ring at the edges)
    waves = {}
    split = {'I': output.w1, 'III+': output.w3_pos, 'V+': output.w5_pos, 'III-': output.w3_neg, 'V-': output.w5_neg}
    if any(w is None for w in split.values()):
        raise ValueError("output has no split waves; run model2018 with 'w' in storeflag")
    for name, w in split.items():
        base = np.mean(w[max(i_click_in - n_edge, 0):max(i_click_in, 1)])
        waves[name] = sps.resample_poly(w - base, up, 1)
    if dur is None:
        dur = 1 / fMod                                  # one period: a single click, each component generated once
    n = int(round(dur * fs))
    t = np.arange(n) / fs
    L_unit = int(round(fs / fMod))                      # one period in samples (whole number after upsampling)
    i_click = i_click_in * up                           # sample of the click after upsampling

    # delta train over the whole signal: 1 at the start of every click, 0 elsewhere
    delta = np.zeros(n)
    if single_click:
        delta[0] = 1
    else:
        delta[::L_unit] = 1
    delta_train = np.vstack([t, delta])

    # unit responses: response of each component to the click, from the click onset
    unit_responses = {}
    for name, w in waves.items():
        L_ur = len(w) - i_click - n_edge * up if window is None else int(round(window * fs))
        if L_ur <= 0 or i_click + L_ur > len(w):
            raise ValueError(f'click_onset + window = {(i_click + L_ur) / fs * 1e3:.1f} ms does not fit in the '
                             f'{len(w) / fs * 1e3:.1f} ms of model output; lower window or make the click epoch longer')
        unit_responses[name] = np.vstack([np.arange(L_ur) / fs, w[i_click:i_click + L_ur]])

    # I, III+, V+ and the two inhibitory parts (III-, V-) share Lat_4
    lat_of = dict(zip(waves, (latencies[0], latencies[1], latencies[2], latencies[3], latencies[3])))

    rate_functions, convolved = {}, {}
    for name, (_, segment) in unit_responses.items():
        lat = lat_of[name]

        # rate function: delta train shifted forward by the latency (impulses pushed past the end are dropped)
        shift = int(round(lat * fs))
        delta_shift = np.zeros(n)
        if shift < n:
            delta_shift[shift:] = delta[:n - shift]
        rate_functions[name] = np.vstack([t, delta_shift])

        # rate function convolved with the unitary response
        convolved[name] = np.vstack([t, np.convolve(delta_shift, segment, mode='full')[:n]])

        i_ext = np.argmax(np.abs(segment))
        print(f"{name}: extreme {segment[i_ext]*1e6:.4f} uV at {i_ext / fs * 1e3:.2f} ms after the click, "
              f"shifted by {lat * 1e3:.2f} ms")

    # Wave_func = sum of the five convolved components
    total = np.vstack([t, sum(convolved[name][1] for name in convolved)])

    if show_plots:
        fig, axes = plt.subplots(4, 1, figsize=(10, 13))
        fig.suptitle(f'ABR convolution model, excitation / inhibition split ({fMod} Hz, Lat_1..4 = '
                     f'{latencies[0]*1e3:.1f}/{latencies[1]*1e3:.1f}/{latencies[2]*1e3:.1f}/{latencies[3]*1e3:.1f} ms)',
                     fontsize=13)

        for i, name in enumerate(unit_responses):
            axes[0].plot(unit_responses[name][0] * 1e3, unit_responses[name][1] * 1e6, color=f'C{i}',
                         linestyle='--' if name.endswith('-') else '-', label=name)
        axes[0].set_xlabel('Time after the click (ms)')
        axes[0].set_ylabel('Amplitude (uV)')
        axes[0].legend()
        axes[0].grid(True)

        for i, name in enumerate(rate_functions):
            spikes = rate_functions[name][0][rate_functions[name][1] == 1]
            spikes = spikes[spikes <= 0.05]
            axes[1].vlines(spikes * 1e3, 1.5 * i, 1.5 * i + 1, color=f'C{i}', label=name)
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_yticks([])
        axes[1].legend(loc='upper right')
        axes[1].grid(True)

        for i, name in enumerate(convolved):
            axes[2].plot(convolved[name][0], convolved[name][1] * 1e6, color=f'C{i}',
                         linestyle='--' if name.endswith('-') else '-', label=name, alpha=0.8)
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude (uV)')
        axes[2].legend()
        axes[2].grid(True)

        axes[3].plot(total[0], total[1] * 1e6, label='Wave_func (sum of the five components)')
        axes[3].set_xlim(0.0,0.04)
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Amplitude (uV)')
        axes[3].legend()
        axes[3].grid(True)

        plt.tight_layout()
        plt.savefig('abr_convolution_split.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✓ Plot saved as 'abr_convolution_split.png'")

    return {'unit_responses': unit_responses, 'delta_train': delta_train,
            'rate_functions': rate_functions, 'convolved': convolved, 'total': total,
            'latencies': tuple(latencies), 'fs': fs}

def frequency_runner(output, freqs=None, latencies=(0.0, 1.7e-3, 4.3e-3, 7e-3), click_onset=0.03, dur=0.04,
                     show_plots=True):
    """
    Runs abr_convolution_split for every click rate in freqs and overlays the resulting Wave_func(t) curves
    (x = time, y = amplitude), one curve per rate. The colour gets darker with increasing rate.
    Every curve is the response to ONE click (single_click=True): the rate only enters through the latencies
    you give per rate, and each response decays over the whole of dur, also past 1/rate. With the same
    latencies for every rate all curves are identical, so give at least one latency per rate (sequence or
    function of the rate).

    Example: frequency_runner(results['output'], latencies=(0, [1e-3]*22, lambda f: 4e-3 + f * 1e-5, 7e-3))

    Parameters:
    -----------
    output : ModelOutput or list containing it
        model2018 output for ONE click (as for abr_convolution_split)
    freqs : array-like or None, optional
        Click rates [Hz]. None = 22 rates from 5 to 110 Hz in steps of 5 Hz (default None)
    latencies : tuple of 4, optional
        (Lat_1, Lat_2, Lat_3, Lat_4) [s] = wave I, excitatory III, excitatory V, inhibitory III and V, the same
        meaning as in abr_convolution_split. Each entry is set separately and can be
        - a number: the same latency for all rates,
        - a sequence with one latency per rate in freqs,
        - a function f -> latency [s]: the latency as a function of the rate.
    click_onset : float, optional
        Time of the click in the stimulus [s] (default 0.03)
    dur : float, optional
        Duration of every curve [s]; the same for all rates, so the curves are comparable (default 0.04)
    show_plots : bool, optional
        Plot the curves and save them as 'frequency_runner.png' (default True)

    Returns:
    --------
    dict with 'freqs' [Hz], 'latencies' (len(freqs) x 4 array [s]), 'results' (the abr_convolution_split
    dict of every rate), 'curves' (len(freqs) x N array of Wave_func [V]) and 't' (time axis [s]). The
    curves are interpolated onto a common 1e5 Hz grid; 'results' holds the originals at their own rates
    """

    freqs = np.linspace(5, 110, 22) if freqs is None else np.asarray(freqs, dtype=float)

    # one latency per rate and component
    lat = np.zeros((len(freqs), 4))
    for k, spec in enumerate(latencies):
        if callable(spec):
            lat[:, k] = [spec(f) for f in freqs]
        elif np.ndim(spec) == 0:
            lat[:, k] = spec
        else:
            if len(spec) != len(freqs):
                raise ValueError(f'latency {k + 1} has {len(spec)} values, expected one per rate ({len(freqs)})')
            lat[:, k] = spec

    results = []
    for f, l in zip(freqs, lat):
        with contextlib.redirect_stdout(io.StringIO()):         # abr_convolution_split prints a line per component
            results.append(abr_convolution_split(output, click_onset=click_onset, fMod=f, dur=dur,
                                                 latencies=tuple(l), show_plots=False, single_click=True))

    if show_plots:
        fig, ax = plt.subplots(figsize=(11, 6))
        cmap = mcolors.LinearSegmentedColormap.from_list('shade', plt.cm.Blues(np.linspace(0.25, 1.0, 256)))
        norm = mcolors.Normalize(freqs.min(), freqs.max())
        for f, res in zip(freqs, results):
            ax.plot(res['total'][0] * 1e3, res['total'][1] * 1e6, color=cmap(norm(f)), linewidth=1.2)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Click rate (Hz)')
        ax.set_xlim(0, dur * 1e3)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (uV)')
        ax.set_title('Wave_func for different click rates')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig('frequency_runner.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✓ Plot saved as 'frequency_runner.png'")

    # every rate has its own upsampled sampling rate (abr_convolution_split), so the curves have different
    # lengths; put them on one common time grid (1e5 Hz) to stack them
    t = np.arange(int(round(dur * 1e5))) / 1e5
    curves = np.array([np.interp(t, r['total'][0], r['total'][1]) for r in results])
    return {'freqs': freqs, 'latencies': lat, 'results': results, 'curves': curves, 't': t}

if __name__ == "__main__":
    # Example usage with different configurations

    print("Running easy model with default parameters...")
    start_time = time.perf_counter()
    results = run_easy_model(stim='click')
    elapsed_time = time.perf_counter() - start_time
    print(f"\nIteration running time: {elapsed_time:.3f} s")

    if results is not None:
        print(f"\nSuccess! EFR value: {results['efr_value']:.4f} μV")
        
        # You can also run with different parameters:
        # results = run_easy_model(carrier_freq=2000, poles_profile='Normal', show_plots=False)
        # results = run_easy_model(carrier_freq=8000, poles_profile='Flat10', save_results=False)
    else:
        print("\nModel run failed. Please check the error messages above.")

    if results is not None:
        abr_results = abr_convolution_split(results['output'], click_onset=0.03, fMod=55, latencies=(0.0, 5.0e-3, 10.0e-3, 15.0e-3))
        frequency_runner(results['output'], freqs=np.linspace(5, 110, 22),
                         latencies=(0.0, lambda f: 1.7e-3 + f * 4e-5, lambda f: 4.3e-3 + f * 8e-5, 7e-3),
                         click_onset=0.03, dur=0.04, show_plots=True)