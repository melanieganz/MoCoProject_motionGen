import numpy as np
import matplotlib.pyplot as plt

# set random seed for replicability
np.random.seed(1)

# this is our old generator
def jakob_gen(
        n_samples=10000,
        seq_len=24,
        dim=6,
        freq_range=(1.0, 5.0),
        amp_range=(0.1, 0.9),
        phase_range=(-np.pi, np.pi)
    ):

    data = list()
    x = np.linspace(0, 2*np.pi, seq_len)
    for n_sample in range(n_samples):
        waves = np.zeros((seq_len, dim))
        rng = np.random.default_rng(seed=n_sample)
        for i in range(dim):
            # all waves have random frequency, amplitude and phase
            frequency = rng.uniform(*freq_range)
            phase = rng.uniform(*phase_range)
            amplitude = rng.uniform(*amp_range)
            wave = amplitude * np.sin(frequency * x + phase) 
            waves[:, i] = wave
        data.append(waves)
    return data


def timegan_gen(
        n_samples=10000,
        x=None,
        seq_len=24,
        dim=6, freq_range=(1.0, 5.0),
        amp_range=None,
        phase_range=(-np.pi, np.pi),
        do_scale=True,
        do_normalize=False,
    ):

    data = list()

    for n_sample in range(n_samples):
        
        temp = list()
        rng = np.random.default_rng(seed=n_sample)

        if x is None:
            x = range(seq_len)

        for n_sine in range(dim):

            freq = rng.uniform(freq_range)
            phase = rng.uniform(*phase_range)
            if amp_range is None:
                amp = 1.
            else:
                amp = np.random.uniform(*amp_range)

            temp_data = [amp * np.sin(freq * x_ + phase) for x_ in x]
            temp.append(temp_data)

        temp = np.transpose(np.asarray(temp))
        if do_scale:
            temp = (temp + 1)*0.5
        if do_normalize:
            temp_flat = temp.flatten()
            temp_min = temp_flat.min()
            temp_max = temp_flat.max()
            temp = (temp - temp_min) / (temp_min - temp_max)
        data.append(temp)

    return data    


# plots a random sample
def plot_sines(data, title):
    _ = plt.figure()
    n, (_, d) = len(data), data[-1].shape
    rng = np.random.default_rng(seed=42)
    idx = rng.randint(0, n)
    for i in range(d):
        plt.plot(data[idx][:, i])
    plt.title(title)
    plt.savefig('./figures/' + title + '.png')