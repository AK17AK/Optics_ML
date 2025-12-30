import numpy as np
import matplotlib.pyplot as plt
from General_GS import generalized_gs


def create_smooth_triangle(size, center, side_length):
    Y, X = np.ogrid[:size, :size]
    cx, cy = center
    # Mathematical definition of an equilateral triangle
    h = side_length * np.sqrt(3) / 2
    # Define three lines that bound the triangle
    c1 = (np.sqrt(3) * (X - cx) - (Y - cy) + h / 3) > 0
    c2 = (-np.sqrt(3) * (X - cx) - (Y - cy) + h / 3) > 0
    c3 = (Y - cy + 2 * h / 3) > 0
    shape = np.logical_and(c1, np.logical_and(c2, c3)).astype(float)
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(shape, sigma=1.0)


def create_smooth_circle(size, center, radius):
    Y, X = np.ogrid[:size, :size]
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    return 1 / (1 + np.exp((dist - radius) / 1.0))


def run_triangle_to_circles(size=256):
    mid = size // 2

    # 1. INPUT PLANE: A Triangle on the left
    input_amp = np.zeros((size, size))
    input_amp[:, :mid] = create_smooth_triangle(size, (mid // 2, mid), side_length=80)[:, :mid]

    total_energy = np.sum(input_amp ** 2)

    # 2. OUTPUT PLANE: 3 Circles on the right
    # Arranged in a small triangular cluster
    target_circles = (create_smooth_circle(size, (mid + 60, mid - 40), radius=15) +
                      create_smooth_circle(size, (mid + 60, mid + 40), radius=15) +
                      create_smooth_circle(size, (mid + 100, mid), radius=15))

    # Normalize Target Circles to 50% energy
    current_target_energy = np.sum(target_circles ** 2)
    target_norm = target_circles * np.sqrt((0.5 * total_energy) / (current_target_energy + 1e-9))

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

        # Target Side: The 3 Circles
        # Control Side: Noise normalized to 50% energy
        control_side_amp = amp * mask_control
        current_control_energy = np.sum(control_side_amp ** 2)
        scale_factor = np.sqrt((0.5 * total_energy) / (current_control_energy + 1e-9))
        control_side_amp_norm = control_side_amp * scale_factor

        final_amp = np.where(mask_target, target_norm, control_side_amp_norm)
        return final_amp * np.exp(1j * phase)

    # 5. Run
    initial_guess = np.exp(1j * np.random.rand(size, size) * 2 * np.pi)
    field_X, errs = generalized_gs(initial_guess, constraint_X, constraint_Y, iterations=500)

    field_Y = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_X)))
    return field_X, field_Y, errs


# Run and Plot
fX, fY, errs = run_triangle_to_circles(256)

plt.figure(figsize=(14, 5))
plt.subplot(1, 3, 1);
plt.title("Input (Triangle)");
plt.imshow(np.abs(fX), cmap='gray')
plt.subplot(1, 3, 2);
plt.title("Output (3 Circles)");
plt.imshow(np.abs(fY), cmap='gray')
plt.subplot(1, 3, 3);
plt.title("Error Log");
plt.plot(errs);
plt.yscale('log')
plt.show()