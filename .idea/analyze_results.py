# analyze_results.py

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load saved data
# -----------------------------
Vc   = np.load("Vc_learned.npy")      # learned control vector
loss = np.load("loss_history.npy")    # loss over epochs
U    = np.load("U.npy")               # same unitary used in training

print("Loaded Vc:", Vc.shape)
print("Loaded loss history:", loss.shape)
print("Loaded U, shape:", U.shape)

# Working set: the 10 images and labels used during training
ws = np.load("working_set.npz")
Vd_work      = ws["Vd_work"]          # shape (10, 784)
labels_work  = ws["labels_work"]      # shape (10,)
print("Loaded working set:", Vd_work.shape, labels_work.shape)

# Prototypes: fixed targets in output plane
pz = np.load("prototypes.npz")
Vu_zero = pz["Vu_zero"]               # complex vector (1024,)
Vu_one  = pz["Vu_one"]                # complex vector (1024,)
print("Loaded prototypes: Vu_zero shape =", Vu_zero.shape, ", Vu_one shape =", Vu_one.shape)

# -----------------------------
# 2. Plot loss vs epoch
# -----------------------------
plt.figure(figsize=(6, 4))
plt.plot(loss, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=200)
plt.show()
print("Saved loss curve to 'loss_curve.png'.")

# -----------------------------
# 3. Define helpers
# -----------------------------
def forward(Vd, Vc):
    """
    Forward pass through the optical system:
    Vo = U @ [Vd; Vc]
    Vd: (784,), Vc: (240,), U: (1024,1024)
    returns: Vo (1024,) complex
    """
    Vdc = np.concatenate([Vd, Vc])     # (1024,)
    return U @ Vdc

def target_for(label):
    """
    Return the fixed prototype for label 0 or 1.
    """
    return Vu_zero if label == 0 else Vu_one

# -----------------------------
# 4. Compare before/after training
# -----------------------------
print("\nDistance to target before vs after training:\n")

Vc_init = np.zeros_like(Vc)           # 'before' = control all zeros (or any baseline you choose)

for i in range(Vd_work.shape[0]):
    Vd  = Vd_work[i]
    lbl = int(labels_work[i])

    Vo_init  = forward(Vd, Vc_init)
    Vo_final = forward(Vd, Vc)

    dist_init  = np.linalg.norm(Vo_init  - target_for(lbl))
    dist_final = np.linalg.norm(Vo_final - target_for(lbl))

    print(f"Sample {i:02d} (label {lbl}) → Before: {dist_init:.4f} | After: {dist_final:.4f}")

print("\nDone. You can now use 'loss_curve.png' and these numbers in your poster/presentation.")
