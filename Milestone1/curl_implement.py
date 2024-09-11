import os
import numpy as np
import matplotlib.pyplot as plt
from muFFT import FFT

# Disable shared memory support for Open MPI
os.environ['OMPI_MCA_btl'] = '^sm'

# Define the grid points
nb_grid_pts = (32, 32, 2)

# Initialize FFT
fft = FFT(nb_grid_pts, engine='pocketfft')

# Function to compute wavenumbers manually
def compute_wavenumbers(nb_grid_pts):
    kx = np.fft.fftfreq(nb_grid_pts[0]).reshape(-1, 1, 1)
    ky = np.fft.fftfreq(nb_grid_pts[1]).reshape(1, -1, 1)
    kz = np.fft.fftfreq(nb_grid_pts[2]).reshape(1, 1, -1)
    return kx, ky, kz

def curl(u_cxyz):
    """Computes the curl of a vector field in real space."""
    u_kxyz = fft.fft(u_cxyz)
    kx, ky, kz = compute_wavenumbers(nb_grid_pts)

    # Ensure wavenumbers have the correct shape
    kx = np.broadcast_to(kx, u_kxyz[0].shape)
    ky = np.broadcast_to(ky, u_kxyz[0].shape)
    kz = np.broadcast_to(kz, u_kxyz[0].shape)

    curl_u_kx = 1j * (ky * u_kxyz[2] - kz * u_kxyz[1])
    curl_u_ky = 1j * (kz * u_kxyz[0] - kx * u_kxyz[2])
    curl_u_kz = 1j * (kx * u_kxyz[1] - ky * u_kxyz[0])

    curl_u_k = np.array([curl_u_kx, curl_u_ky, curl_u_kz])
    curl_u_c = fft.ifft(curl_u_k).real

    return curl_u_c

# Task 3: Test that a constant field has vanishing curl
u_cxyz = np.ones([3, *fft.nb_subdomain_grid_pts])
curlu_cxyz = curl(u_cxyz)
np.testing.assert_allclose(curlu_cxyz, 0)

# Task 4: Test nonvanishing curl
norm = np.array([0, 0, 1])
u_cxyz = np.cross(norm, fft.coords - 0.5, axis=0)

# Compute the curl
curlu_cxyz = curl(u_cxyz)

# Plot vector field and curl of the vector field
X, Y = np.meshgrid(np.arange(nb_grid_pts[0]), np.arange(nb_grid_pts[1]))
U, V = u_cxyz[0, ..., 0], u_cxyz[1, ..., 0]
CU, CV = curlu_cxyz[0, ..., 0], curlu_cxyz[1, ..., 0]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].quiver(X, Y, U, V)
ax[0].set_title('Vector Field')

ax[1].quiver(X, Y, CU, CV)
ax[1].set_title('Curl of the Vector Field')

plt.show()