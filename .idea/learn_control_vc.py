# learn_control_vc.py
import os
import numpy as np
import pandas as pd

# ----------------------------
# Utilities
# ----------------------------
def random_unitary(n, rng=np.random.default_rng(0)):
    """Generate a random unitary matrix via complex QR with phase-fixing."""
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    # Ensure diag(R) has positive real to make Q unique up to global phase
    d = np.diag(R)
    ph = d / np.abs(d)
    Q = Q * ph
    return Q  # Q^H Q = I

def is_unitary(U, tol=1e-8):
    I = np.eye(U.shape[0], dtype=U.dtype)
    err = np.linalg.norm(U.conj().T @ U - I) / U.shape[0]
    return err, err < tol

def load_mnist_0_1_from_csv(csv_path, n_per_class_proto=500, n_working=10, seed=42):
    """
    Load MNIST from CSV (Kaggle-style: label in col 0, 784 pixels in cols 1:).
    Returns:
      Vd_work (10x784), labels_work (10,), zeros_proto, ones_proto (each up to n_per_class_proto x 784)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None)
    df01 = df[(df.iloc[:,0]==0) | (df.iloc[:,0]==1)].copy()

    # Working set: 10 samples for the learning loop (your original choice)
    work = df01.sample(n=n_working, random_state=seed)
    labels_work = work.iloc[:,0].to_numpy()
    Vd_work = work.iloc[:,1:].to_numpy(dtype=np.float64) / 255.0  # normalize to [0,1]

    zeros = df01[df01.iloc[:,0]==0].iloc[:,1:].to_numpy(dtype=np.float64) / 255.0
    ones  = df01[df01.iloc[:,0]==1].iloc[:,1:].to_numpy(dtype=np.float64) / 255.0
    # Prototypes subset to control runtime (adjust if you want full class)
    rng = np.random.default_rng(seed)
    if len(zeros) > n_per_class_proto:
        zeros = zeros[rng.choice(len(zeros), size=n_per_class_proto, replace=False)]
    if len(ones) > n_per_class_proto:
        ones  = ones[rng.choice(len(ones),  size=n_per_class_proto, replace=False)]

    return Vd_work, labels_work, zeros, ones

def build_prototypes(U, zeros_proto, ones_proto, Vc_proto):
    """
    Build fixed targets in the OUTPUT plane using a fixed control Vc_proto (no drift).
    Returns complex vectors Vu_zero, Vu_one (length 1024).
    """
    n = U.shape[0]
    assert n == 1024, "This script assumes 784+240=1024."
    def avg_out(X):
        acc = np.zeros(n, dtype=np.complex128)
        for x in X:
            vdc = np.concatenate([x, Vc_proto])
            acc += U @ vdc
        return acc / len(X)

    Vu_zero = avg_out(zeros_proto)
    Vu_one  = avg_out(ones_proto)
    return Vu_zero, Vu_one

# ----------------------------
# Losses / gradients
# ----------------------------
def complex_field_residual(Vo, Vu):
    """Residual in complex field (simple L2 on complex vector)."""
    return Vo - Vu  # complex residual

def amplitude_residual(Vo, Au, eps=1e-12):
    """
    Residual that matches amplitude target Au by keeping complex direction of Vo.
    r = (|Vo| - Au) * Vo / (|Vo| + eps)
    """
    mag = np.abs(Vo)
    return (mag - Au) * (Vo / (mag + eps))

# ----------------------------
# Main training routine
# ----------------------------
def main():
    # ---- Settings ----
    CSV = "mnist_train.csv"  # change if needed
    LOSS = "amplitude"           # "field" or "amplitude"
    SAME_CLASS_TARGET = True # True: push Vo -> same-class prototype; False: opposite-class (not recommended)
    LR = 1e-2
    EPOCHS = 20
    BATCH_SIZE = 10          # we use your 10-sample working set as one batch
    SEED = 42

    rng = np.random.default_rng(SEED)

    # ---- Load data ----
    Vd_work, labels_work, zeros_proto, ones_proto = load_mnist_0_1_from_csv(
        CSV, n_per_class_proto=500, n_working=BATCH_SIZE, seed=SEED
    )

    # ---- Build U (unitary) ----
    U = random_unitary(1024, rng)  # or replace with your GetRandomMatrix.generate_unitary_matrix(1024)
    err, ok = is_unitary(U)
    print(f"[check] ||U^H U - I||/N = {err:.2e}  (unitary_ok={ok})")

    # SAVE U for analysis
    np.save("U.npy", U)

    # ---- Fixed prototype control (no drift in targets) ----
    Vc_proto = np.zeros(240, dtype=np.float64)  # or rng.uniform(0,1,240)
    Vu_zero, Vu_one = build_prototypes(U, zeros_proto, ones_proto, Vc_proto)

    # SAVE prototypes for analysis
    np.savez("prototypes.npz", Vu_zero=Vu_zero, Vu_one=Vu_one)

    # SAVE working set for analysis
    np.savez("working_set.npz", Vd_work=Vd_work, labels_work=labels_work)

    # If using amplitude loss, precompute prototype amplitudes:
    if LOSS == "amplitude":
        Au_zero = np.abs(Vu_zero)
        Au_one  = np.abs(Vu_one)

    # ---- Learn the shared control Vc ----
    Vc = rng.uniform(0.0, 1.0, size=240).astype(np.float64)

    hist = []
    for epoch in range(1, EPOCHS+1):
        # Batch gradient
        grad_c = np.zeros_like(Vc)
        loss_acc = 0.0

        # Process the 10-sample batch
        for i in range(BATCH_SIZE):
            label = int(labels_work[i])
            Vd = Vd_work[i]  # shape (784,), [0,1]

            # Choose target
            if SAME_CLASS_TARGET:
                Vu = Vu_zero if label == 0 else Vu_one
            else:
                Vu = Vu_one if label == 0 else Vu_zero  # (usually not what you want)

            # Forward
            Vdc = np.concatenate([Vd, Vc])           # (1024,)
            Vo  = U @ Vdc                             # (1024,) complex

            # Residual
            if LOSS == "field":
                r = complex_field_residual(Vo, Vu)    # complex
            elif LOSS == "amplitude":
                Au = Au_zero if label == 0 else Au_one
                r  = amplitude_residual(Vo, Au)       # complex (directional)
            else:
                raise ValueError("LOSS must be 'field' or 'amplitude'.")

            # Loss value (scalar) just for logging
            if LOSS == "field":
                loss_acc += 0.5 * np.vdot(r, r).real / len(r)
            else:  # amplitude
                loss_acc += 0.5 * np.mean((np.abs(Vo) - (Au_zero if label==0 else Au_one))**2)

            # Backprop to input domain: grad = U^H r
            grad = U.conj().T @ r                     # complex
            grad_c += grad[-240:].real                # real control: take real part

        # Step
        Vc -= (LR / BATCH_SIZE) * grad_c
        # Project to physical range (amplitude SLM-like)
        Vc = np.clip(Vc, 0.0, 1.0)

        avg_loss = loss_acc / BATCH_SIZE
        hist.append(avg_loss)
        print(f"[epoch {epoch:02d}] loss={avg_loss:.6e}")

    print("\nTraining complete.")
    print("Final control vector stats: min={:.4f}, mean={:.4f}, max={:.4f}"
          .format(Vc.min(), Vc.mean(), Vc.max()))

    # Save Vc and loss history
    np.save("Vc_learned.npy", Vc)
    np.save("loss_history.npy", np.array(hist, dtype=np.float64))

if __name__ == "__main__":
    main()
