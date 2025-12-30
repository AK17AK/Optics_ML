import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from General_GS import generalized_gs


def create_smooth_triangle(size, center, side_length, rotation=0):
    Y, X = np.ogrid[:size, :size]
    cx, cy = center
    h = side_length * np.sqrt(3) / 2
    if rotation == 180:
        c1, c2, c3 = (np.sqrt(3) * (X - cx) - (cy - Y) + h / 3) > 0, (-np.sqrt(3) * (X - cx) - (cy - Y) + h / 3) > 0, (
                    cy - Y + 2 * h / 3) > 0
    else:
        c1, c2, c3 = (np.sqrt(3) * (X - cx) - (Y - cy) + h / 3) > 0, (-np.sqrt(3) * (X - cx) - (Y - cy) + h / 3) > 0, (
                    Y - cy + 2 * h / 3) > 0
    shape = np.logical_and(c1, np.logical_and(c2, c3)).astype(float)
    return gaussian_filter(shape, sigma=1.0)


def create_smooth_square(size, center, side_length):
    shape = np.zeros((size, size))
    cx, cy = center
    half = side_length // 2
    shape[cy - half:cy + half, cx - half:cx + half] = 1.0
    return gaussian_filter(shape, sigma=0.5)


def run_squares_to_triangles_split(size=256):
    mid = size // 2
    # System energy budget
    base_energy = 10000

    # 1. INPUT PLANE: 4 Squares on the LEFT
    input_target_shapes = np.zeros((size, size))
    # Note: placing centers in the left half (0 to mid)
    sq_centers = [(mid // 4, mid // 2), (3 * mid // 4, mid // 2), (mid // 2, mid // 4), (mid // 2, 3 * mid // 4)]
    for center in sq_centers:
        input_target_shapes += create_smooth_square(size, center, side_length=20)

    input_target_norm = input_target_shapes * np.sqrt((0.5 * base_energy) / (np.sum(input_target_shapes ** 2) + 1e-9))

    # 2. OUTPUT PLANE: 2 Triangles on the RIGHT
    output_target_shapes = (create_smooth_triangle(size, (mid + mid // 2, mid - 50), side_length=60) +
                            create_smooth_triangle(size, (mid + mid // 2, mid + 50), side_length=60, rotation=180))

    output_target_norm = output_target_shapes * np.sqrt(
        (0.5 * base_energy) / (np.sum(output_target_shapes ** 2) + 1e-9))

    # 3. MASKS
    # Input Plane: Left = Target, Right = Control
    mask_in_target = np.zeros((size, size), dtype=bool)
    mask_in_target[:, :mid] = True
    mask_in_control = ~mask_in_target

    # Output Plane: Left = Control, Right = Target
    mask_out_target = np.zeros((size, size), dtype=bool)
    mask_out_target[:, mid:] = True
    mask_out_control = ~mask_out_target

    # 4. Constraints
    def constraint_X(field):
        amp = np.abs(field)
        phase = np.angle(field)

        # Right side is free control
        control_amp = amp * mask_in_control
        scale = np.sqrt((0.5 * base_energy) / (np.sum(control_amp ** 2) + 1e-9))

        # Left side is the 4 squares
        final_amp = np.where(mask_in_target, input_target_norm, control_amp * scale)
        return final_amp * np.exp(1j * phase)

    def constraint_Y(field):
        amp = np.abs(field)
        phase = np.angle(field)

        # Left side is free control (energy dump)
        control_amp = amp * mask_out_control
        scale = np.sqrt((0.5 * base_energy) / (np.sum(control_amp ** 2) + 1e-9))

        # Right side is the 2 triangles
        final_amp = np.where(mask_out_target, output_target_norm, control_amp * scale)
        return final_amp * np.exp(1j * phase)

    # 5. Run
    initial_guess = (np.random.rand(size, size) + 1j * np.random.rand(size, size))
    field_X, errs = generalized_gs(initial_guess, constraint_X, constraint_Y, iterations=100)

    field_Y = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_X)))
    return field_X, field_Y, errs


# Execute and Plot
fX, fY, errs = run_squares_to_triangles_split(256)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Input Plane (Squares L | Control R)")
plt.imshow(np.abs(fX), cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Output Plane (Control L | Triangles R)")
plt.imshow(np.abs(fY), cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Error Log")
plt.plot(errs)
plt.yscale('log')
plt.show()