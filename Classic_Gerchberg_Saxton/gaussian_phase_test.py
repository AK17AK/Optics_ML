import numpy as np
import matplotlib.pyplot as plt
from gerchberg_saxton import gerchberg_saxton

# Grid
N = 256
x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x)
R = np.sqrt(X**2 + Y**2)  # Radial coordinate

aperture_radius = 0.8
I_input = (R < aperture_radius).astype(float)
A_input = np.sqrt(I_input)

#Define the Target Phase (The phase we want to retrieve)
#2D Gaussian function
sigma = 0.3
true_phase = -np.pi * np.exp(-R**2 / (2 * sigma**2))

# Create the TRUE Complex Input Field
field_input_true = A_input * np.exp(1j * true_phase)

#Compute the Target Intensity (I_target)
# This is the far-field intensity *required* by the true phase
field_target_true = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_input_true)))
I_target = np.abs(field_target_true)**2
I_target /= I_target.max()  # Normalize target intensity

# --- RUN GERCHBERG-SAXTON ---

# Use I_input (uniform illumination) and I_target (calculated far-field pattern)
# Fewer iterations are often needed for continuous functions
field_reconstructed, phase_reconstructed = gerchberg_saxton(I_input, I_target, num_iterations=500)

# --- ANALYSIS AND PLOTTING ---

#Compute phase difference (wrapped to -π..π)
# We expect the difference to be a constant offset + error
# Note: GS can only retrieve the phase within the illuminated region (R < aperture_radius)
phase_diff = np.angle(np.exp(1j * (phase_reconstructed - true_phase)))

#Compute statistics inside the aperture only

#Plots
plt.figure(figsize=(16, 4))

plt.subplot(141)
plt.imshow(A_input, cmap='gray')
plt.title("Input Amplitude (Aperture)")

plt.subplot(142)
plt.imshow(I_target, cmap='viridis') # Use 'viridis' for intensity visualization
plt.title("Target Intensity (Far-Field)")

plt.subplot(143)
plt.imshow(phase_reconstructed, cmap='hsv') # 'hsv' is great for phase visualization
plt.title(r"Retrieved Phase $\phi_{rec}$")

plt.subplot(144)
plt.imshow(phase_diff, cmap='gray', vmin=-np.pi, vmax=np.pi)
plt.title(r"Phase Difference $\phi_{rec} - \phi_{true}$")
plt.colorbar(label='Radians', fraction=0.046)

plt.tight_layout()
plt.show()