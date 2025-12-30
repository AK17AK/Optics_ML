import numpy as np
import matplotlib.pyplot as plt
from General_GS import generalized_gs


def create_smooth_circle(size, center, radius):
    Y, X = np.ogrid[:size, :size]
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    return 1 / (1 + np.exp((dist - radius) / 1.0))


def create_smooth_square(size, top_left, side_length):
    shape = np.zeros((size, size))
    r, c = top_left
    shape[r:r + side_length, c:c + side_length] = 1.0
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(shape, sigma=0.5)


def run_split_energy_transform(size=256):
    mid = size // 2

    # 1. Define Input: A clean circle centered in the frame
    # (Note: Usually input is centered, but we can put it on the left if you prefer.
    # For standard GS, a centered input source is best.)
    input_amp = create_smooth_circle(size, (mid, mid), radius=40)

    # Calculate Total Energy Available in the System
    total_energy = np.sum(input_amp ** 2)

    # 2. Define Output Target: Squares on the RIGHT side
    target_squares = np.zeros((size, size))
    # Add squares to the right half
    sqs = (create_smooth_square(size, (80, mid + 40), 30) +
           create_smooth_square(size, (150, mid + 80), 30))
    target_squares = sqs

    # Normalize Target Squares to have exactly 50% of Total Energy
    current_sq_energy = np.sum(target_squares ** 2)
    target_squares_norm = target_squares * np.sqrt((0.5 * total_energy) / (current_sq_energy + 1e-9))

    # 3. Define the Split Masks
    # Left Half = Control | Right Half = Target
    mask_control = np.zeros((size, size), dtype=bool)
    mask_control[:, :mid] = True  # Left Half

    mask_target = np.zeros((size, size), dtype=bool)
    mask_target[:, mid:] = True  # Right Half

    # 4. Constraints
    def constraint_X(field):
        # Input Plane: Force the Laser/Circle Amplitude, Keep the Phase
        # This is standard. We don't change the laser source, only its phase.
        current_phase = np.angle(field)
        return input_amp * np.exp(1j * current_phase)

    def constraint_Y(field):
        # Output Plane: The "Split Energy" Logic
        amp = np.abs(field)
        phase = np.angle(field)

        # A. HANDLE TARGET SIDE (Right)
        # We strictly enforce the shape of the squares
        target_side_amp = target_squares_norm

        # B. HANDLE CONTROL SIDE (Left)
        # We let the pattern be whatever it wants (to solve the phase),
        # BUT we normalize it to have exactly 50% of the energy.
        control_side_amp = amp * mask_control  # Extract current random noise on left
        current_control_energy = np.sum(control_side_amp ** 2)

        # Scaling factor to force exactly 50% energy
        scale_factor = np.sqrt((0.5 * total_energy) / (current_control_energy + 1e-9))
        control_side_amp_norm = control_side_amp * scale_factor

        # Combine them
        # (Right side is the strict squares, Left side is the normalized noise)
        final_amp = np.where(mask_target, target_squares_norm, control_side_amp_norm)

        return final_amp * np.exp(1j * phase)

    # 5. Run Generalized GS
    initial_guess = np.exp(1j * np.random.rand(size, size) * 2 * np.pi)
    field_X, errs = generalized_gs(initial_guess, constraint_X, constraint_Y, iterations=1000)

    field_Y = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_X)))
    return field_X, field_Y, errs


# Run and Plot
fX, fY, errs = run_split_energy_transform(256)

plt.figure(figsize=(14, 5))

# Plot 1: The Input Phase (The Hologram we made)
plt.subplot(1, 3, 1)
plt.title("Input Plane (Circle Source)")
plt.imshow(np.abs(fX), cmap='gray')

# Plot 2: The Output (Split Energy)
plt.subplot(1, 3, 2)
plt.title("Output: Control(L) vs Target(R)")
# We use vmax to ensure we aren't clipping the brightness
plt.imshow(np.abs(fY), cmap='gray')

# Plot 3: Error
plt.subplot(1, 3, 3)
plt.title("Convergence Error")
plt.plot(errs)
plt.yscale('log')

plt.show()