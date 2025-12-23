import numpy as np
import matplotlib.pyplot as plt
from General_GS import generalized_gs


# ==========================================
# 1. SYNTHETIC SHAPE GENERATORS
# ==========================================
def create_circle(size, center, radius):
    Y, X = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    return (dist_from_center <= radius).astype(float)


def create_square(size, top_left, side_length):
    shape = np.zeros((size, size))
    r, c = top_left
    shape[r:r + side_length, c:c + side_length] = 1.0
    return shape


# ==========================================
# 2. EXPERIMENT SETUP
# ==========================================
def run_synthetic_transform(size=256):
    mid = size // 2

    # --- A. Create Target X (Left Half: Circle) ---
    target_X = np.zeros((size, size))
    # Place a circle in the center of the LEFT half
    circle = create_circle(size, (mid // 2, mid), radius=40)
    target_X[:, :mid] = circle[:, :mid]

    # --- B. Create Target Y (Right Half: Three Squares) ---
    target_Y = np.zeros((size, size))
    # Place three squares in the RIGHT half
    sq1 = create_square(size, (mid - 60, mid + 40), 30)
    #sq2 = create_square(size, (mid + 20, mid + 20), 30)
    #sq3 = create_square(size, (mid - 10, mid + 70), 30)
    #target_Y[:, mid:] = (sq1 + sq2 + sq3)[:, mid:]
    target_Y[:, mid:] = (sq1)[:, mid:]

    # --- C. Define Masks ---
    mask_X = np.zeros((size, size), dtype=bool)
    mask_X[:, :mid] = True

    mask_Y = np.zeros((size, size), dtype=bool)
    mask_Y[:, mid:] = True

    # --- D. Define Constraints ---
    def constraint_X(field):
        out = field.copy()
        # Enforce the Circle amplitude on the left, keep phase
        out[mask_X] = target_X[mask_X] * np.exp(1j * np.angle(out[mask_X]))
        return out

    def constraint_Y(field):
        # Enforce the Squares amplitude on the right, keep phase
        out = field.copy()
        out[mask_Y] = target_Y[mask_Y] * np.exp(1j * np.angle(out[mask_Y]))
        return out

    # --- E. Run ---
    initial_guess = np.random.rand(size, size) + 1j * np.random.rand(size, size)

    # Geometric shapes usually need fewer iterations to show clear results
    final_field_X, errs = generalized_gs(initial_guess, constraint_X, constraint_Y, iterations=1000)

    final_field_Y = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(final_field_X)))

    return final_field_X, final_field_Y, errs, target_X, target_Y


# ==========================================
# 3. VISUALIZATION
# ==========================================
size = 256
field_X, field_Y, errs, target_X, target_Y = run_synthetic_transform(size)

plt.figure(figsize=(15, 10))

# ROW 1: Input Domain (X)
plt.subplot(2, 3, 1)
plt.title("1. Target X (Circle)")
plt.imshow(target_X, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("2. Actual Input Amplitude")
plt.imshow(np.abs(field_X), cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("3. Input Phase (The Hologram)")
plt.imshow(np.angle(field_X), cmap='twilight')
plt.axis('off')

# ROW 2: Output Domain (Y)
plt.subplot(2, 3, 4)
plt.title("4. Target Y (Squares)")
plt.imshow(target_Y, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("5. Output Result (Linear)")
plt.imshow(np.abs(field_Y), cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("6. Output Result (Log Scale)")
output_amp = np.abs(field_Y)
plt.imshow(np.log(output_amp + 1), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Convergence plot
plt.figure()
plt.plot(errs)
plt.title("Convergence Error")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.show()