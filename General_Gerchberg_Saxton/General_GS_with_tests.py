# The code implements a generalized Gerchberg-Saxton, and runs two tests.
# __________________________________________________________________________________
# Inpainting removes ~60% of the image (0 light) at the input domain
# the amplitude of the field in the frequency domain is known, the phase is unknown.
# __________________________________________________________________________________
# Split removes the right half of the image in the input domain
# and removes the left half of the image in the frequency domain
# phase information isn't removed here.
# __________________________________________________________________________________
# This can be generalized to other cool combinations, and can be tested on more complex images with complex phases
# The tests here involve a circle with two asymmetrical holes, the phase of the image is zero.
# we can add a complex phase, a more complex input image...
import numpy as np
import matplotlib.pyplot as plt


def generalized_gs(initial_guess, constraint_func_X, constraint_func_Y, iterations=100):
    """
    The algorithm moves back and forth between domains, applying custom constraints.
    """
    field = initial_guess.astype(complex)
    errors = []

    for i in range(iterations):
        # 1. Domain X to Y (Space to Frequency)
        field_f = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))

        # 2. Apply Constraint in Y
        # We save state to calculate error
        prev_field_f = field_f.copy()
        field_f = constraint_func_Y(field_f)

        # Track convergence (how much did the constraint change the field?)
        errors.append(np.mean(np.abs(prev_field_f - field_f) ** 2))

        # 3. Domain Y to X (Frequency to Space)
        field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field_f)))

        # 4. Apply Constraint in X
        field = constraint_func_X(field)

    return field, errors


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
    recovered, errs = generalized_gs(initial_guess, constraint_X, constraint_Y, iterations=500)

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
    recovered, errs = generalized_gs(initial_guess, constraint_X, constraint_Y, iterations=500)

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