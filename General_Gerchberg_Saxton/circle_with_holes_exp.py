import numpy as np
import matplotlib.pyplot as plt
from General_GS import generalized_gs
# Helper to create a sample image (A Circle with 2 holes)
def create_target_image(size=128):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    target = (X ** 2 + Y ** 2 < 0.5).astype(float)
    target[50:60, 30:40] = 0  # Add a hole somewhere
    target[70:95, 80:95] = 0 # Add another hole somewhere else
    return target


# ==========================================
# SCENARIO 1: Holographic Inpainting (Random Drop)
# ==========================================
# We know 40 of the image (random) in the 1st domain
# We know the Amplitude of the image in the frequency domain (phase is unknown)
def run_inpainting_experiment(target_img):
    size = target_img.shape[0]

    # 1. Generate Constraints
    # We know the Full Amplitude in Frequency Domain (Standard GS)
    target_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(target_img)))
    target_amp_Y = np.abs(target_fft)

    # We only know 40% of the pixels in the Spatial Domain
    mask_X = np.random.rand(size, size) > 0.6
    known_X_values = target_img * mask_X

    # 2. Define Constraint Functions
    def constraint_X(field):
        # Enforce known pixels, let unknown pixels evolve
        out = field.copy()
        # Enforcing only the amplitude in X (since target_img is real)
        out[mask_X] = target_img[mask_X] * np.exp(1j * np.angle(out[mask_X]))
        # A simpler approach (assuming a real image is being recovered):
        # out[mask_X] = known_X_values[mask_X]
        return out

    def constraint_Y(field):
        # Enforce Amplitude, keep Phase
        return target_amp_Y * np.exp(1j * np.angle(field))

    # 3. Run
    print("Running Inpainting Experiment...")
    initial_guess = np.random.rand(size, size) + 1j * np.random.rand(size, size)
    recovered, errs = generalized_gs(initial_guess, constraint_X, constraint_Y, iterations=5000)

    return recovered, mask_X, errs


# ==========================================
# SCENARIO 2: The "Split-Brain" (Left/Right Split)
# ==========================================
def run_split_experiment(target_img):
    size = target_img.shape[0]
    mid = size // 2

    # Truth Data
    target_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(target_img)))

    # --- Setup Masks ---

    # Plane X Mask: We know the LEFT half
    mask_X = np.zeros((size, size), dtype=bool)
    mask_X[:, :mid] = True
    known_X_values = target_img.copy()

    # Plane Y Mask: We know the RIGHT half of the Diffraction Pattern
    mask_Y = np.zeros((size, size), dtype=bool)
    mask_Y[:, mid:] = True
    known_Y_values = target_fft.copy()

    # --- Define Constraint Functions ---

    def constraint_X(field):
        # Enforce Left Half, let Right Half evolve
        out = field.copy()
        # Enforcing only the amplitude in X (since target_img is real)
        out[mask_X] = known_X_values[mask_X] * np.exp(1j * np.angle(out[mask_X]))
        return out

    def constraint_Y(field):
        # Enforce Right Half of Spectrum, let Left Half evolve
        out = field.copy()
        out[mask_Y] = known_Y_values[mask_Y]
        return out

    # --- Run ---
    print("Running Split-Brain Experiment...")
    # Start with random noise
    initial_guess = np.random.rand(size, size) + 1j * np.random.rand(size, size)
    recovered, errs = generalized_gs(initial_guess, constraint_X, constraint_Y, iterations=1500)

    return recovered, errs


# ==========================================
# Visualization
# ==========================================
target = create_target_image()

# Run 1
rec_1, mask_1, err_1 = run_inpainting_experiment(target)
# Run 2
rec_2, err_2 = run_split_experiment(target)

plt.figure(figsize=(15, 8)) # Increased size for better viewing

# --- Row 1: Original Image and Inpainting Results ---

plt.subplot(2, 4, 1)
plt.title("0. Original Target Image")
plt.imshow(target, cmap='gray')

plt.subplot(2, 4, 2)
plt.title("1. Inpainting Input (40% known)")
masked_view = target.copy()
masked_view[~mask_1] = 0  # Black out the unknown pixels for display
plt.imshow(masked_view, cmap='gray')

plt.subplot(2, 4, 3)
plt.title("2. Recovered Inpainting Image")
plt.imshow(np.abs(rec_1), cmap='gray')

plt.subplot(2, 4, 4)
plt.title("3. Convg. Error (Inpainting)")
plt.plot(err_1)
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('MSE (Log Scale)')


# --- Row 2: Split-Brain Results ---

plt.subplot(2, 4, 5)
plt.title("4. Split Input X (Left Known)")
split_view = target.copy()
split_view[:, 64:] = 0
plt.imshow(split_view, cmap='gray')

plt.subplot(2, 4, 6)
plt.title("5. Recovered Split Image")
plt.imshow(np.abs(rec_2), cmap='gray')

plt.subplot(2, 4, 7)
plt.title("6. Convg. Error (Split)")
plt.plot(err_2)
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('MSE (Log Scale)')

# Placeholder to balance the subplot layout
plt.subplot(2, 4, 8)
plt.title("Comparison Placeholder")
plt.text(0.5, 0.5, 'Compare 2 & 5 with 0',
         horizontalalignment='center', verticalalignment='center',
         fontsize=12, color='darkgray')
plt.axis('off')


plt.tight_layout()
plt.show()