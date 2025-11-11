import numpy as np
import matplotlib.pyplot as plt
from gerchberg_saxton import gerchberg_saxton

# Grid
N = 256
x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x)
R = np.sqrt(X**2 + Y**2)  # Radial coordinate
#Define input amplitude: single vertical slit
slit_width = 0.3
I_input = (np.abs(X) < slit_width/2).astype(float)

#Compute its far-field intensity (target)
#    The Fourier transform of a slit is a sinc pattern
field_input = np.sqrt(I_input)
field_target = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_input)))
I_target = np.abs(field_target)**2
I_target /= I_target.max()  # normalize

field_reconstructed, phase_reconstructed = gerchberg_saxton(I_input, I_target, 100)
true_phase = np.zeros_like(phase_reconstructed)

#Compute phase difference (wrapped to -π..π)
phase_diff = np.angle(np.exp(1j * (phase_reconstructed - true_phase)))

# Plots
plt.figure(figsize=(16,4))
plt.subplot(141); plt.imshow(I_input, cmap='gray'); plt.title("Input (slit)")
plt.subplot(142); plt.imshow(I_target, cmap='gray'); plt.title("Target (sinc² pattern)")
plt.subplot(143); plt.imshow(phase_reconstructed, cmap='gray'); plt.title("Retrieved phase")
plt.subplot(144); plt.imshow(phase_diff, cmap='gray', vmin=-np.pi, vmax=np.pi)
plt.title("Phase difference (rec - true)")
plt.colorbar(fraction=0.046)
plt.tight_layout()
plt.show()
