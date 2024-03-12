import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel

def generate_smooth_signal(num_samples, num_waves, gamma=1.0):
    waves = np.zeros((num_samples, num_waves))

    for i in range(num_waves):
        # Generate grid of equally-spaced points
        x = np.linspace(0, 10, num_samples)
    
        # Define RBF kernel
        K = rbf_kernel(x.reshape(-1, 1), gamma=gamma)
        wave = np.random.multivariate_normal(mean=np.zeros(num_samples), cov=K)
        waves[:, i] = wave

    return waves

# Number of samples to generate
num_samples_to_generate = 10000
# Number of sample points to generate
sample_points = 100
# Number of waves per sample
num_waves = 1

smooth_funcs = np.zeros((num_samples_to_generate, sample_points, num_waves))

# Generate sine wave samples
for i in range(num_samples_to_generate):
    smooth_funcs[i] = generate_smooth_signal(sample_points, num_waves)

np.savez('custom_smooths.npz', data=smooth_funcs)

# Plot the first sample from the array
plt.figure(figsize=(12, 8))
for i in range(num_waves):
    plt.plot(smooth_funcs[0, :, i], label=f'Smooth func. {i+1}')

plt.title('Generated Smooth Function')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()