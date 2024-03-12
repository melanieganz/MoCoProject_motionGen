import numpy as np
import matplotlib.pyplot as plt

# Function to generate multiple sine waves as one sample
def generate_multiple_sine_waves(num_samples, num_waves, frequency_range, amplitude_range):
    waves = np.zeros((num_samples, num_waves))

    for i in range(num_waves):
        frequency = np.random.uniform(*frequency_range)
        amplitude = np.random.uniform(*amplitude_range)
        phase = np.random.uniform(-np.pi, np.pi)

        x = np.linspace(0, 2*np.pi, num_samples)
        wave = amplitude * np.sin(frequency * x + phase)
        waves[:, i] = wave

    return waves

# Number of sample points to generate
sample_points = 24
# Number of waves per sample
num_waves = 6
# Frequency range for generated sine waves
frequency_range = (1.0, 5.0)
# Amplitude range for generated sine waves
amplitude_range = (0.1, 0.9)
# Number of samples to generate
num_samples_to_generate = 10000

# Initialize an array to store the sine waves
sine_wave_samples = np.zeros((num_samples_to_generate, sample_points, num_waves))

# Generate sine wave samples
for i in range(num_samples_to_generate):
    sine_wave_samples[i] = generate_multiple_sine_waves(sample_points, num_waves, frequency_range, amplitude_range)

np.savez('custom_sines.npz', data=sine_wave_samples)

# Plot the first sample from the array
plt.figure(figsize=(12, 8))
for i in range(num_waves):
    plt.plot(sine_wave_samples[0, :, i], label=f'Sine Wave {i+1}')

plt.title('Generated Sine Wave Sample')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
