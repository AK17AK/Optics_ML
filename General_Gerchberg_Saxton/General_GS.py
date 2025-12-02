# The code implements a generalized Gerchberg-Saxton, and runs two tests.
# __________________________________________________________________________________
# Inpainting removes ~60% of the image (0 light) at the input domain
# the amplitude of the field in the frequency domain is known, the phase is unknown.
# __________________________________________________________________________________
# Split removes the right half of the image in the input domain
# and removes the left half of the image in the frequency domain
# phase information isn't removed here.
# __________________________________________________________________________________
# This can be generalized to other cool combinations, and can be tested on more complex images with complex phases
# The tests here involve a circle with two asymmetrical holes, the phase of the image is zero.
# we can add a complex phase, a more complex input image...
import numpy as np


def generalized_gs(initial_guess, constraint_func_X, constraint_func_Y, iterations=100):
    """
    The algorithm moves back and forth between domains, applying custom constraints.
    """
    field = initial_guess.astype(complex)
    errors = []

    for i in range(iterations):
        # 1. Domain X to Y (Space to Frequency)
        field_f = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))

        # 2. Apply Constraint in Y
        # We save state to calculate error
        prev_field_f = field_f.copy()
        field_f = constraint_func_Y(field_f)

        # Track convergence (how much did the constraint change the field?)
        errors.append(np.mean(np.abs(prev_field_f - field_f) ** 2))

        # 3. Domain Y to X (Frequency to Space)
        field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field_f)))

        # 4. Apply Constraint in X
        field = constraint_func_X(field)
        if i % 100 == 0:
            print(f"Iter {i}: Error {errors[-1]:.6f}")

    return field, errors
