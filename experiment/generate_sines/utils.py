import numpy as np
import matplotlib.pyplot as plt

# set random seed for replicability
np.random.seed(1)

# this is our old generator
def jakob_gen(no=10000, seq_len=24, dim=6, freq_range=(1.0, 5.0), amp_range=(0.1, 0.9), phase_range=(-np.pi, np.pi)):
    data = list()
    x = np.linspace(0, 2*np.pi, seq_len)
    for _ in range(no):
        waves = np.zeros((seq_len, dim))
        for i in range(dim):
            # all waves have random frequency, amplitude and phase
            frequency = np.random.uniform(*freq_range)
            amplitude = np.random.uniform(*amp_range)
            phase = np.random.uniform(*phase_range)
            wave = amplitude * np.sin(frequency * x + phase) 
            waves[:, i] = wave
        data.append(waves)
    return data
    
# this is the TimeGAN generator (no modifications)
def timegan_gen_1(no=10000, seq_len=24, dim=6, freq_range=(1.0, 5.0), amp_range=(0.1, 0.9), phase_range=(-np.pi, np.pi)):
    data = list()
    for _ in range(no):
        temp = list()
        for _ in range(dim):
            # all waves have random frequency and phase
            freq = np.random.uniform(*freq_range)
            phase = np.random.uniform(*phase_range)
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)
        temp = np.transpose(np.asarray(temp))
        temp = (temp + 1)*0.5
        data.append(temp)
    return data    

# this is the TimeGAN generator (modified x-axis)
def timegan_gen_2(no=10000, seq_len=24, dim=6, freq_range=(1.0, 5.0), amp_range=(0.1, 0.9), phase_range=(-np.pi, np.pi)):
    data = list()
    x = np.linspace(0, 2*np.pi, seq_len)
    for _ in range(no):
        temp = list()
        for _ in range(dim):
            # all waves have random frequency and phase
            freq = np.random.uniform(*freq_range)
            phase = np.random.uniform(*phase_range)
            temp_data = np.sin(freq * x + phase)
            temp.append(temp_data)
        temp = np.transpose(np.asarray(temp))
        temp = (temp + 1)*0.5
        data.append(temp)
    return data

# this is the TimeGAN generator (modified x-axis, random amplitude per wave)
def timegan_gen_3(no=10000, seq_len=24, dim=6, freq_range=(1.0, 5.0), amp_range=(0.1, 0.9), phase_range=(-np.pi, np.pi)):
    data = list()
    x = np.linspace(0, 2*np.pi, seq_len)
    for _ in range(no):
        temp = list()
        for _ in range(dim):
            # all waves have random frequency, amplitude and phase
            freq = np.random.uniform(*freq_range)
            amp = np.random.uniform(*amp_range)
            phase = np.random.uniform(*phase_range)
            temp_data = amp * np.sin(freq * x + phase)
            temp.append(temp_data)
        temp = np.transpose(np.asarray(temp))
        data.append(temp)
    return data

# this is the TimeGAN generator (modified x-axis, random amplitude across samples, similar within each sample)
def timegan_gen_4(no=10000, seq_len=24, dim=6, freq_range=(1.0, 5.0), amp_range=(0.1, 0.9), phase_range=(-np.pi, np.pi)):
    data = list()
    x = np.linspace(0, 2*np.pi, seq_len)
    for _ in range(no):
        temp = list()
        for _ in range(dim):
            # all waves have random frequency and phase
            freq = np.random.uniform(*freq_range)
            phase = np.random.uniform(*phase_range)
            temp_data = np.sin(freq * x + phase)
            temp.append(temp_data)
        temp = np.transpose(np.asarray(temp))
        amp = np.random.uniform(*amp_range)
        temp = temp * amp
        data.append(temp)
    return data

# plots a random sample
def plot_sines(data, title):
    _ = plt.figure()
    n, (_, d) = len(data), data[-1].shape
    idx = np.random.randint(0, n)
    for i in range(d):
        plt.plot(data[idx][:, i])
    plt.title(title)
    plt.savefig('./figures/' + title + '.png')