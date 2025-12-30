import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from General_GS import generalized_gs


def create_smooth_triangle(size, center, side_length, rotation=0):
    Y, X = np.ogrid[:size, :size]
    cx, cy = center
    # Basic equilateral triangle math
    h = side_length * np.sqrt(3) / 2

    # Rotation logic (simple 180 flip if needed)
    if rotation == 180:
        c1 = (np.sqrt(3) * (X - cx) - (cy - Y) + h / 3) > 0
        c2 = (-np.sqrt(3) * (X - cx) - (cy - Y) + h / 3) > 0
        c3 = (cy - Y + 2 * h / 3) > 0
    else:
        c1 = (np.sqrt(3) * (X - cx) - (Y - cy) + h / 3) > 0
        c2 = (-np.sqrt(3) * (X - cx) - (Y - cy) + h / 3) > 0
        c3 = (Y - cy + 2 * h / 3) > 0

    shape = np.logical_and(c1, np.logical_and(c2, c3)).astype(float)
    return gaussian_filter(shape, sigma=1.0)


def create_smooth_square(size, center, side_length):
    shape = np.zeros((size, size))
    cx, cy = center
    half = side_length // 2
    shape[cy - half:cy + half, cx - half:cx + half] = 1.0
    return gaussian_filter(shape, sigma=0.5)


def run_squares_to_triangles(size=256):
    mid = size // 2

    # 1. INPUT PLANE: 4 Squares on the left
    input_amp = np.zeros((size, size))
    sq_centers = [(mid // 4, mid // 2), (3 * mid // 4, mid // 2), (mid // 2, mid // 4), (mid // 2, 3 * mid // 4)]
    for center in sq_centers:
        input_amp += create_smooth_square(size, center, side_length=20)

    # Restrict input to the left half
    input_amp[:, mid:] = 0
    total_energy = np.sum(input_amp ** 2)

    # 2. OUTPUT PLANE: 2 Triangles on the right
    target_triangles = (create_smooth_triangle(size, (mid + mid // 2, mid - 50), side_length=60) +
                        create_smooth_triangle(size, (mid + mid // 2, mid + 50), side_length=60, rotation=180))

    # Normalize Target Triangles to 50% energy
    current_target_energy = np.sum(target_triangles ** 2)
    target_norm = target_triangles * np.sqrt((0.5 * total_energy) / (current_target_energy + 1e-9))

    # 3. Split Masks
    mask_control = np.zeros((size, size), dtype=bool)
    mask_control[:, :mid] = True

    mask_target = np.zeros((size, size), dtype=bool)
    mask_target[:, mid:] = True

    # 4. Constraints
    def constraint_X(field):
        current_phase = np.angle(field)
        return input_amp * np.exp(1j * current_phase)

    def constraint_Y(field):
        amp = np.abs(field)
        phase = np.angle(field)

        # Target Side: The 2 Triangles
        # Control Side: Noise normalized to 50% energy
        control_side_amp = amp * mask_control
        current_control_energy = np.sum(control_side_amp ** 2)
        scale_factor = np.sqrt((0.5 * total_energy) / (current_control_energy + 1e-9))
        control_side_amp_norm = control_side_amp * scale_factor

        final_amp = np.where(mask_target, target_norm, control_side_amp_norm)
        return final_amp * np.exp(1j * phase)

    # 5. Run
    initial_guess = np.exp(1j * np.random.rand(size, size) * 2 * np.pi)
    field_X, errs = generalized_gs(initial_guess, constraint_X, constraint_Y, iterations=900)

    field_Y = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_X)))
    return field_X, field_Y, errs


# Execute
fX, fY, errs = run_squares_to_triangles(256)

# Visualization
plt.figure(figsize=(14, 5))
plt.subplot(1, 3, 1);
plt.title("Input (4 Squares)");
plt.imshow(np.abs(fX), cmap='gray')
plt.subplot(1, 3, 2);
plt.title("Output (2 Triangles)");
plt.imshow(np.abs(fY), cmap='gray')
plt.subplot(1, 3, 3);
plt.title("Error Log");
plt.plot(errs);
plt.yscale('log')
plt.show()