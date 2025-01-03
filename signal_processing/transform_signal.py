import numpy as np
import scipy

def mel_filter_bank(sr,n_fft,n_mels,fmin,fmax):
    """
    Create a Mel filter bank to convert a spectrogram to a Mel-spectrogram

    Parameters:
    - sr: Sample rate
    - n_fft: Length of FFT windown
    - n_mels: Number of Mel bands
    - fmin: Minimum frequency (Hz)
    - fmax: Maximum frequency (Hz)

    Returns:
    - Mel filter bank as numpy array
    """
    def hz_to_mel(hz):
        return 2595*np.log10(1+hz/700)

    def mel_to_hz(mel):
        return 700*(10**(mel/2595)-1)
        
    # Compute the Mel filter bank
    mel_points = np.linspace(hz_to_mel(fmin),hz_to_mel(fmax),n_mels+2)
    hz_points = mel_to_hz(mel_points)

    #FFT bin frequncies

    bin_points = np.floor((n_fft+1)*hz_points/sr).astype(int)

    #Create the Mel filter bank
    filter_bank = np.zeros((n_mels,int(n_fft//2+1)))
    for i in range(1, n_mels + 1):
        filter_bank[i - 1, bin_points[i - 1]:bin_points[i]] = np.linspace(0, 1, bin_points[i] - bin_points[i - 1])
        filter_bank[i - 1, bin_points[i]:bin_points[i + 1]] = np.linspace(1, 0, bin_points[i + 1] - bin_points[i])
    
    return filter_bank

def extract_stft_feature(signal, sfreq, time_seg):
    f,t,Zxx = scipy.signal.stft(x=signal,fs=sfreq,nperseg=int(time_seg*sfreq))
    magnitude = np.abs(Zxx)
    return f,t,magnitude

def extract_mel_spectrogram(signal_data,sfreq=256,time_seg=1,n_mels=128,
                            fmin=0.5,fmax=None):
    """
    Compute Mel-spectrogram using scipy for a given signal.
    
    Parameters:
    - signal_data: 1D numpy array of the audio signal
    - sr: Sample rate of the audio data
    - n_fft: Number of FFT points
    - hop_length: Number of samples between successive frames
    - n_mels: Number of Mel bands
    - fmin: Minimum frequency for the Mel filter bank (Hz)
    - fmax: Maximum frequency for the Mel filter bank (Hz)
    
    Returns:
    - Mel-spectrogram (dB scale)
    """
    if fmax is None:
        fmax = float(sfreq //2)
    f,t,magnitude = extract_stft_feature(signal = signal_data,sfreq=sfreq,time_seg =time_seg)
    n_fft_bins = len(f) 
    mel_filter = mel_filter_bank(sfreq, n_fft=n_fft_bins*2, n_mels=n_mels, fmin=fmin, fmax=fmax)

    mel_spectrogram = np.dot(mel_filter[:, :n_fft_bins], magnitude)
    mel_spectrogram_db = 10 * np.log10(np.maximum(mel_spectrogram, 1e-10))
    return mel_spectrogram_db, t

def extract_allsignal(signal, time_seg=0.5, n_mels=128, fmin=0.5, fmax=None):
    """
    Compute Mel-spectrograms for all channels of the EEG signal.

    Parameters:
    - signal: mne.io.Raw object containing the EEG data
    - time_seg: Segment duration in seconds
    - n_mels: Number of Mel bands
    - fmin: Minimum frequency for the Mel filter bank (Hz)
    - fmax: Maximum frequency for the Mel filter bank (Hz)

    Returns:
    - mel_spectrograms: numpy array of shape (channels, mel_bins, time_steps)
    """
    mel_spectrograms = []

    # Get data and apply a Hann window
    data, _ = signal[:]
    data *= scipy.signal.windows.hann(data.shape[1])

    # Compute Mel-spectrogram for each channel
    for i, channel_name in enumerate(signal.ch_names):
        mel_spectrogram, _ = extract_mel_spectrogram(
            signal_data=data[i],
            sfreq=signal.info['sfreq'],
            time_seg=time_seg,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )
        mel_spectrograms.append(mel_spectrogram)

    mel_spectrograms = np.array(mel_spectrograms)
    return mel_spectrograms



