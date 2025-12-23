import numpy as np
import matplotlib.pyplot as plt
from General_GS import generalized_gs


def create_smooth_circle(size, center, radius):
    Y, X = np.ogrid[:size, :size]
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    # Using a soft edge (sigmoid-like) helps Fourier convergence
    return 1 / (1 + np.exp((dist - radius) / 1.0))


def create_smooth_square(size, top_left, side_length):
    shape = np.zeros((size, size))
    r, c = top_left
    shape[r:r + side_length, c:c + side_length] = 1.0
    # A tiny bit of blur helps
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(shape, sigma=0.5)


def run_balanced_transform(size=256):
    mid = size // 2

    # 1. Create Targets
    target_X_raw = np.zeros((size, size))
    target_X_raw[:, :mid] = create_smooth_circle(size, (mid // 2, mid), radius=50)[:, :mid]

    target_Y_raw = np.zeros((size, size))
    sqs = (create_smooth_square(size, (80, mid + 40), 40) +
           create_smooth_square(size, (150, mid + 80), 30))
    target_Y_raw[:, mid:] = sqs[:, mid:]

    # 2. ENERGY BALANCING (The Fix)
    # Scale Target Y so it has the same total energy as Target X
    energy_X = np.sum(target_X_raw ** 2)
    energy_Y = np.sum(target_Y_raw ** 2)
    target_Y_balanced = target_Y_raw * np.sqrt(energy_X / (energy_Y + 1e-9))

    mask_X = np.zeros((size, size), dtype=bool)
    mask_X[:, :mid] = True
    mask_Y = np.zeros((size, size), dtype=bool)
    mask_Y[:, mid:] = True

    # 3. Constraints
    def constraint_X(field):
        out = field.copy()
        # Enforce Amplitude, Keep Phase (Standard GS)
        out[mask_X] = target_X_raw[mask_X] * np.exp(1j * np.angle(out[mask_X]))
        return out

    def constraint_Y(field):
        out = field.copy()
        # Enforce Balanced Amplitude, Keep Phase
        out[mask_Y] = target_Y_balanced[mask_Y] * np.exp(1j * np.angle(out[mask_Y]))
        return out

    # 4. Run
    initial_guess = np.exp(1j * np.random.rand(size, size) * 2 * np.pi)
    field_X, errs = generalized_gs(initial_guess, constraint_X, constraint_Y, iterations=1500)

    field_Y = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_X)))
    return field_X, field_Y, errs, target_X_raw, target_Y_balanced


# Run and Plot
field_X, field_Y, errs, tX, tY = run_balanced_transform(256)

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1);
plt.title("Input (Circle Side)");
plt.imshow(np.abs(field_X), cmap='gray')
plt.subplot(1, 3, 2);
plt.title("Output (Squares Side)");
plt.imshow(np.log(np.abs(field_Y)), cmap='gray')

plt.subplot(1, 3, 3);
plt.title("Error Log");
plt.plot(errs);
plt.yscale('log')

plt.show()