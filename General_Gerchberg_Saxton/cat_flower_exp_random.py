## this currently just stagnates at a local minima with a huge error, this needs to be solved
import numpy as np
import matplotlib.pyplot as plt
from General_GS import generalized_gs
import requests
from PIL import Image
from io import BytesIO


# ==========================================
# 1. IMAGE LOADER (kept as is)
# ==========================================
def load_and_prep_image(url, target_shape):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Download failed: {url}")
    try:
        img = Image.open(BytesIO(response.content)).convert('L')
    except Exception as e:
        raise Exception(f"PIL Error on {url}") from e

    img = img.resize((target_shape[1], target_shape[0]))
    img_array = np.array(img).astype(float)
    img_array /= 255.0
    return img_array

# ==========================================
# 2. EXPERIMENT: Sparse Control (Random 50%)
# ==========================================
def run_sparse_control_experiment(size=256):
    # --- A. Load Images ---
    url_cat = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/400px-Cat_November_2010-1a.jpg"
    url_flower = "https://upload.wikimedia.org/wikipedia/commons/1/1c/Pink_twinged_daisy_on_table_edit.jpg"

    print("Downloading images...")
    cat_img = load_and_prep_image(url_cat, (size, size))
    flower_img = load_and_prep_image(url_flower, (size, size))

    # --- B. Create TWO Independent Random Masks ---
    #np.random.seed(42)  # Seed for the Cat mask (M_X)
    mask_X = np.random.rand(size, size) > 0.6

    #np.random.seed(99)  # DIFFERENT Seed for the Flower mask (M_Y)
    mask_Y = np.random.rand(size, size) > 0.6

    # Pre-calculate the fixed reference value for efficiency (Amplitude + Zero Phase)
    # These references are used inside the constraints to enforce immutability.
    fixed_ref_X = cat_img * np.exp(1j * 0)
    fixed_ref_Y = flower_img * np.exp(1j * 0)

    # --- C. Define Constraints ---

    def constraint_X(field):
        """
        Input Plane: Fix 50% of pixels to the Cat's value (Amplitude + Phase=0).
        """
        out = field.copy()
        # Enforce the fixed reference complex value on the M_X mask
        out[mask_X] = fixed_ref_X[mask_X]
        return out

    def constraint_Y(field):
        """
        Output Plane: Fix a DIFFERENT 50% of pixels to the Flower's value (Amplitude + Phase=0).
        """
        out = field.copy()
        # Enforce the fixed reference complex value on the M_Y mask
        out[mask_Y] = fixed_ref_Y[mask_Y]
        return out

    # --- D. Run ---
    initial_guess = np.random.rand(size, size) + 1j * np.random.rand(size, size)

    # CRITICAL FIX: Increased iterations for convergence in rigid, double-sided problem
    field_X, errs = generalized_gs(initial_guess, constraint_X, constraint_Y, iterations=500)

    field_Y = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_X)))

    # Return all necessary variables, including the masks
    return field_X, field_Y, errs, cat_img, flower_img, mask_X, mask_Y, constraint_X, constraint_Y


# ==========================================
# 3. VISUALIZATION (Fixed Clipping for Visibility)
# ==========================================

size = 256
# Run the experiment (runs with 8000 iterations now)
results = run_sparse_control_experiment(size)
field_X, field_Y, errs, cat_img, flower_img, mask_X, mask_Y, constraint_X, constraint_Y = results

plt.figure(figsize=(18, 9))

# --- Row 1: The Setup ---
plt.subplot(2, 4, 1)
plt.title("Constraint X Mask ($\mathbf{M}_X$)\n(Fixed Cat Pixels)")
plt.imshow(mask_X, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.title("Constraint Y Mask ($\mathbf{M}_Y$)\n(Fixed Flower Pixels)")
plt.imshow(mask_Y, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.title("Target X (Cat)")
plt.imshow(cat_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.title("Target Y (Flower)")
plt.imshow(flower_img, cmap='gray')
plt.axis('off')

# --- Row 2: The Results (Using Clipping to Guarantee Visibility) ---
plt.subplot(2, 4, 5)
plt.title("ACTUAL Input X Amplitude\n(Cat Amplitude Enforced)")
# FIX: Clip the amplitude display to 1.0 to ensure the Cat is visible
clipped_amplitude_X = np.clip(np.abs(field_X), 0, 1.0)
plt.imshow(clipped_amplitude_X, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.title("ACTUAL Output Y Amplitude\n(Reconstructed Flower)")
# FIX: Clip the output to a high percentile to see the Flower over the noise
clipped_amplitude_Y = np.clip(np.abs(field_Y), 0, np.percentile(np.abs(field_Y), 99.9))
plt.imshow(clipped_amplitude_Y, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.title("Convergence Error")
plt.plot(errs)
plt.yscale('log')
plt.xlabel("Iterations")
plt.grid(True, alpha=0.3)

plt.subplot(2, 4, 8)
plt.title("Control Phase (Plane X)")
plt.imshow(np.angle(field_X), cmap='twilight')
plt.axis('off')

plt.tight_layout()
plt.show()

# ==========================================
# 4. TWO-ITERATION TEST
# ==========================================

print("\n\n--- Two-Iteration Verification Test ---")

# Start field is the final converged field_X from the main run
field_X0 = field_X.copy()

# --- ITERATION 1 (X -> Y -> X') ---
print("Iteration 1: X -> Y -> X'")

# 1. Forward Transform (X -> Y)
field_Y1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_X0)))

# 2. Apply Y Constraint
field_Y1_c = constraint_Y(field_Y1)

# 3. Inverse Transform (Y' -> X')
field_X1_c = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field_Y1_c)))

# 4. Apply X Constraint
field_X1_final = constraint_X(field_X1_c)


# --- ITERATION 2 (X' -> Y'' -> X'') ---
print("Iteration 2: X' -> Y'' -> X''")

# 1. Forward Transform (X' -> Y'')
field_Y2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_X1_final)))

# 2. Apply Y Constraint
field_Y2_c = constraint_Y(field_Y2)

# 3. Inverse Transform (Y'' -> X'')
field_X2_c = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field_Y2_c)))

# 4. Apply X Constraint
field_X2_final = constraint_X(field_X2_c)


# --- CALCULATE DIFFERENCE ---
# The primary stagnation check is the difference between the field after Iteration 1 and Iteration 2.
# If this difference is near zero, the process is stuck.
diff_amp = np.mean(np.abs(np.abs(field_X2_final) - np.abs(field_X1_final)))
diff_phase = np.mean(np.abs(np.angle(field_X2_final) - np.angle(field_X1_final)))
total_diff = np.mean(np.abs(field_X2_final - field_X1_final))

print(f"\nMean Absolute Difference (Amplitude): {diff_amp:.6e}")
print(f"Mean Absolute Difference (Phase):   {diff_phase:.6e}")
print(f"Total Complex Difference:           {total_diff:.6e}")

if total_diff < 1e-6:
    print("STATUS: STAGNATION The field is no longer changing significantly between iterations.")
else:
    print("STATUS: The field is still evolving/bouncing between iterations.")