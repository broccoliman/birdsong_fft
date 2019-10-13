from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np


def diff440(x: float) -> float:
    '''
    half note difference from 440 (probably only works if x > 440)
    x - frequency
    returns - difference in halfnotes from an a note (440)
    '''
    n =  x=np.log2(x/440)*12
    return n - int(n)//12*12

def freq_max(arr: np.ndarray, sr: int) -> float:
    '''
    Frequency with highest amplitude from arr, arr.shape = (n,)

    arr - wav file as np.ndarray, arr.shape = (n,) = mono
    sr - samplerate, int

    returns - frequency of highest amplitude
    '''
    spec = np.abs(np.fft.rfft(arr))
    freq = np.fft.rfftfreq(len(arr), d=1 / sr)
    spec = np.abs(spec)
    amp = spec / spec.sum()
    mode = freq[amp.argmax()]
    return mode


def freq_median(arr: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(arr))
    freq = np.fft.rfftfreq(len(arr), d=1 / sr)
    spec = np.abs(spec)
    amp = spec / spec.sum()
    mean = (freq * amp).sum()
    sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    amp_cumsum = np.cumsum(amp)
    median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
    return median


def sw(n, dn, arr, sr, use_median=False):
    '''
    sliding window freqency check

    n - window size
    dn - distance between window positions
    arr - wav file as np.ndarray
    sr - samplerate
    use_median - sliding window returns median frequency instead of max amplitude one
    
    returns - time, freq, amp
    '''

    if use_median:
        freq = freq_median
    else:
        freq = freq_max

    l = (len(arr)-n)//dn-1

    data = np.zeros([l,3])
    for i in range(l):
        segm = arr[i*dn : i*dn + n]
        data[i,0] = i*dn/sr
        data[i,1] = freq(segm, sr)
        data[i,2] = np.abs(segm).mean()
    return data[:,0], data[:,1], data[:,2]


def plot_interval():
    sr, sound = wavfile.read("yellow_warbler1.wav")
    sound = sound[:,0].astype(np.float64)

    t,freq,vol = sw(10000, 20, sound, sr, True)

    bitmask = vol>100
    freq = freq*bitmask

    indices = freq>440
    notes = [diff440(x) for x in freq[indices]]

    plt.plot(t[indices], notes, '.')
    plt.xlabel('time (s)')
    plt.ylabel('frequency (half notes)')
    plt.title('Half notes above an A note (modulus some octaves)')
    plt.show()

# TODO: play around with parameters n,nd


if __name__ == "__main__":
    plot_interval()