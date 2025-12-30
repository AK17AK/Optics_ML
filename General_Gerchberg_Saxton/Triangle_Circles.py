import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from General_GS import generalized_gs


def create_smooth_triangle(size, center, side_length):
    Y, X = np.ogrid[:size, :size]
    cx, cy = center
    h = side_length * np.sqrt(3) / 2
    c1 = (np.sqrt(3) * (X - cx) - (Y - cy) + h / 3) > 0
    c2 = (-np.sqrt(3) * (X - cx) - (Y - cy) + h / 3) > 0
    c3 = (Y - cy + 2 * h / 3) > 0
    shape = np.logical_and(c1, np.logical_and(c2, c3)).astype(float)
    return gaussian_filter(shape, sigma=1.0)


def create_smooth_circle(size, center, radius):
    Y, X = np.ogrid[:size, :size]
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    return 1 / (1 + np.exp((dist - radius) / 1.0))


def run_swapped_input_gs(size=256):
    mid = size // 2
    base_energy = 10000

    # 1. INPUT DEFINITION (Target now on the LEFT Half)
    input_target = np.zeros((size, size))
    # Placing the triangle in the left quadrant
    input_target[:, :mid] = create_smooth_triangle(size, (mid // 2, mid), side_length=80)[:, :mid]
    input_target_norm = input_target * np.sqrt((0.5 * base_energy) / (np.sum(input_target ** 2) + 1e-9))

    # 2. OUTPUT DEFINITION (Target stays on the RIGHT Half)
    output_target = (create_smooth_circle(size, (mid + 60, mid - 40), radius=15) +
                     create_smooth_circle(size, (mid + 60, mid + 40), radius=15) +
                     create_smooth_circle(size, (mid + 100, mid), radius=15))
    output_target_norm = output_target * np.sqrt((0.5 * base_energy) / (np.sum(output_target ** 2) + 1e-9))

    # 3. MASKS
    # For Input: Left is Target, Right is Control
    mask_input_target = np.zeros((size, size), dtype=bool)
    mask_input_target[:, :mid] = True
    mask_input_control = np.zeros((size, size), dtype=bool)
    mask_input_control[:, mid:] = True

    # For Output: Left is Control, Right is Target
    mask_output_control = np.zeros((size, size), dtype=bool)
    mask_output_control[:, :mid] = True
    mask_output_target = np.zeros((size, size), dtype=bool)
    mask_output_target[:, mid:] = True

    # 4. Constraints
    def constraint_X(field):
        amp = np.abs(field)
        phase = np.angle(field)

        # RIGHT side of Input is now the FREE Control region
        control_amp = amp * mask_input_control
        scale = np.sqrt((0.5 * base_energy) / (np.sum(control_amp ** 2) + 1e-9))

        # LEFT side is the Triangle
        final_amp = np.where(mask_input_target, input_target_norm, control_amp * scale)
        return final_amp * np.exp(1j * phase)

    def constraint_Y(field):
        amp = np.abs(field)
        phase = np.angle(field)

        # LEFT side of Output is the Control/Dump region
        control_amp = amp * mask_output_control
        scale = np.sqrt((0.5 * base_energy) / (np.sum(control_amp ** 2) + 1e-9))

        # RIGHT side is the 3 Circles
        final_amp = np.where(mask_output_target, output_target_norm, control_amp * scale)
        return final_amp * np.exp(1j * phase)

    # 5. Run
    initial_guess = (np.random.rand(size, size) + 1j * np.random.rand(size, size))
    field_X, errs = generalized_gs(initial_guess, constraint_X, constraint_Y, iterations=100)

    field_Y = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_X)))
    return field_X, field_Y, errs


# Execute and Plot
fX, fY, errs = run_swapped_input_gs(256)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Input Plane (Triangle L | Control R)")
plt.imshow(np.abs(fX), cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Output Plane (Control L | Circles R)")
plt.imshow(np.abs(fY), cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Error Log")
plt.plot(errs)
plt.yscale('log')
plt.show()