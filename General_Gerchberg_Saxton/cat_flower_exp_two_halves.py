# Works but isn't very clear
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from General_GS import generalized_gs

# ==========================================
# 1. HELPER: Download & Process Images
# ==========================================
def load_and_prep_image(url, target_shape):
    """
    Downloads an image using a User-Agent header for better compatibility,
    converts to grayscale, and resizes it.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # Use headers in the request
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"Failed to download image from {url}. Status code: {response.status_code}")

    # The fix: Ensure the content is treated as an image stream
    try:
        img = Image.open(BytesIO(response.content)).convert('L')  # Convert to Grayscale
    except Exception as e:
        # If PIL fails, it often means the content was HTML, not an image.
        raise Exception(f"PIL could not identify image file from URL: {url}. (Content may be HTML/redirect).") from e

    img = img.resize((target_shape[1], target_shape[0]))  # PIL uses (width, height)
    img_array = np.array(img).astype(float)
    img_array /= 255.0  # Normalize 0-1
    return img_array




# ==========================================
# 3. SCENARIO: The "Cat-to-Flower" Transformer
# ==========================================
def run_real_image_transform(size=256):  # Use 256 for better resolution
    mid = size // 2

    # --- A. Load Images from Web ---
    # URLs for a standard Cat and a Sunflower
    url_cat = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/400px-Cat_November_2010-1a.jpg"
    url_flower = "https://upload.wikimedia.org/wikipedia/commons/1/1c/Pink_twinged_daisy_on_table_edit.jpg"

    print("Downloading images...")
    # Resize cat to fit the LEFT half (size, mid)
    cat_img = load_and_prep_image(url_cat, (size, mid))

    # Resize flower to fit the RIGHT half (size, mid)
    flower_img = load_and_prep_image(url_flower, (size, mid))

    # --- B. Create The Constraints ---

    # Constraint X Target: Left Half = Cat, Right Half = 0 (placeholder)
    target_X = np.zeros((size, size))
    target_X[:, :mid] = cat_img

    # Constraint Y Target: Left Half = 0 (placeholder), Right Half = Flower
    target_Y = np.zeros((size, size))
    target_Y[:, mid:] = flower_img

    # Define Masks (Where do we enforce the image?)
    mask_X = np.zeros((size, size), dtype=bool)
    mask_X[:, :mid] = True  # We only enforce the Left side of X

    mask_Y = np.zeros((size, size), dtype=bool)
    mask_Y[:, mid:] = True  # We only enforce the Right side of Y

    # --- C. Define Functions ---

    def constraint_X(field):
        # Goal: Force the Left Half to look like the Cat.
        # The Right Half is free to change (it becomes the "holographic code").
        out = field.copy()
        out[mask_X] = target_X[mask_X] * np.exp(1j * np.angle(out[mask_X]))
        return out

    def constraint_Y(field):
        # Goal: Force the Right Half of the SPECTRUM to look like the Flower.
        # The Left Half is free (waste energy / high order diffraction).
        out = field.copy()
        out[mask_Y] = target_Y[mask_Y] * np.exp(1j * np.angle(out[mask_Y]))
        return out

    # --- D. Run ---
    # We use more iterations because complex photos are harder to converge than circles
    initial_guess = np.random.rand(size, size) + 1j * np.random.rand(size, size)

    final_field_X, errs = generalized_gs(initial_guess, constraint_X, constraint_Y, iterations=2000)

    # Compute the final result in Y to verify
    final_field_Y = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(final_field_X)))

    return final_field_X, final_field_Y, errs, cat_img, flower_img


# ==========================================
# 4. VISUALIZATION
# ==========================================

# Run the experiment
size = 256
field_X, field_Y, errs, cat, flower = run_real_image_transform(size)

plt.figure(figsize=(15, 10))

# --- ROW 1: The Input (Plane X) ---
plt.subplot(2, 3, 1)
plt.title("1. The Input Constraint\n(We forced the Left to be Cat)")
plt.imshow(cat, cmap='gray') # Show the pure target
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("2. The Input Result (Amplitude)\n(Left=Cat, Right=Control/Hologram)")
plt.imshow(np.abs(field_X), cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("3. Input Phase (The Secret Code)\n(This creates the transformation)")
plt.imshow(np.angle(field_X), cmap='twilight') # Phase is colorful
plt.axis('off')

# --- ROW 2: The Output (Plane Y - Fourier) ---
plt.subplot(2, 3, 4)
plt.title("4. The Output Constraint\n(We forced Right to be Flower)")
# We create a dummy image to show what we *asked* the algorithm to do
target_display = np.zeros((size, size))
target_display[:, size//2:] = flower
plt.imshow(target_display, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("5. The Actual Output (Linear Scale)\n(Look closely! It might look dark)")
# This is what you were seeing before - looks black because of bright noise on left
plt.imshow(np.abs(field_Y), cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("6. The Actual Output (Log Scale)\n(Proof: The Flower is there!)")
# We use Logarithm to see dark details next to bright noise
output_amp = np.abs(field_Y)
plt.imshow(np.log(output_amp + 1), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()