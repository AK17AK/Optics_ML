import numpy as np


def gerchberg_saxton(I_input, I_target, num_iterations=50):

    # Convert intensities to amplitude
    A_input = np.sqrt(I_input)
    A_target = np.sqrt(I_target)

    # Initialize random phase in input plane
    phase = np.exp(1j * 2 * np.pi * np.random.rand(*A_input.shape))

    # Initial complex field
    field = A_input * phase

    for i in range(num_iterations):
        # Forward Fourier Transform to target plane
        field_f = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))

        # Replace amplitude in target plane
        field_f = A_target * np.exp(1j * np.angle(field_f))

        # Inverse Fourier Transform back to input plane
        field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field_f)))

        # Replace amplitude in input plane
        field = A_input * np.exp(1j * np.angle(field))

    return field, np.angle(field)
