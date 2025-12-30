import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from General_GS import generalized_gs


def create_smooth_circle(size, center, radius):
    Y, X = np.ogrid[:size, :size]
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    return 1 / (1 + np.exp((dist - radius) / 1.0))


def create_smooth_square(size, top_left, side_length):
    shape = np.zeros((size, size))
    r, c = top_left
    shape[r:r + side_length, c:c + side_length] = 1.0
    return gaussian_filter(shape, sigma=0.5)


def run_balanced_split_planes(size=256):
    mid = size // 2
    # Define a base energy budget for the whole system
    base_energy = 10000

    # 1. INPUT DEFINITION (Target on the LEFT Half)
    input_target_raw = np.zeros((size, size))
    # Placing the circle center in the left half
    input_target_raw[:, :mid] = create_smooth_circle(size, (mid // 2, mid), radius=40)[:, :mid]

    # Normalize input target to 50% energy
    input_target_norm = input_target_raw * np.sqrt((0.5 * base_energy) / (np.sum(input_target_raw ** 2) + 1e-9))

    # 2. OUTPUT DEFINITION (Target on the RIGHT Half)
    target_squares_raw = np.zeros((size, size))
    sqs = (create_smooth_square(size, (80, mid + 40), 30) +
           create_smooth_square(size, (150, mid + 80), 30))
    target_squares_raw[:, mid:] = sqs[:, mid:]

    # Normalize output target to 50% energy
    output_target_norm = target_squares_raw * np.sqrt((0.5 * base_energy) / (np.sum(target_squares_raw ** 2) + 1e-9))

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

        # RIGHT side is free to be whatever it wants (Control)
        control_amp = amp * mask_in_control
        scale = np.sqrt((0.5 * base_energy) / (np.sum(control_amp ** 2) + 1e-9))

        # LEFT side is strictly the Circle
        final_amp = np.where(mask_in_target, input_target_norm, control_amp * scale)
        return final_amp * np.exp(1j * phase)

    def constraint_Y(field):
        amp = np.abs(field)
        phase = np.angle(field)

        # LEFT side is free control (Energy dump)
        control_amp = amp * mask_out_control
        scale = np.sqrt((0.5 * base_energy) / (np.sum(control_amp ** 2) + 1e-9))

        # RIGHT side is strictly the Squares
        final_amp = np.where(mask_out_target, output_target_norm, control_amp * scale)
        return final_amp * np.exp(1j * phase)

    # 5. Run
    # Random guess with some amplitude to help the first normalization step
    initial_guess = (np.random.rand(size, size) + 1j * np.random.rand(size, size))
    field_X, errs = generalized_gs(initial_guess, constraint_X, constraint_Y, iterations=100)

    field_Y = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_X)))
    return field_X, field_Y, errs


# Execute and Plot
fX, fY, errs = run_balanced_split_planes(256)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Input Plane (Circle L | Control R)")
plt.imshow(np.abs(fX), cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Output Plane (Control L | Squares R)")
plt.imshow(np.abs(fY), cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Error Log")
plt.plot(errs)
plt.yscale('log')
plt.show()