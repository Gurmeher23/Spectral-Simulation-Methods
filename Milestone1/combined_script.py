import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import muFFT
import muGrid

class FFT:
    def __init__(self, nb_grid_pts, engine='pocketfft'):
        self.nb_grid_pts = nb_grid_pts
        self.engine = engine
        self.communicator = muGrid.Communicator()
        self.fft = muFFT.FFT(nb_grid_pts, engine=self.engine, communicator=self.communicator)
    
    def forward(self, u):
        rfield = self.fft.real_space_field('rfield')
        print(f"Forward: input real-space field shape: {u.shape}")
        rfield.p = u
        ffield = self.fft.fourier_space_field('ffield')
        self.fft.fft(rfield, ffield)
        print(f"Forward: output Fourier-space field shape: {ffield.p.shape}")
        return ffield.p
    
    def backward(self, u_hat):
        ffield = self.fft.fourier_space_field('ffield')
        print(f"Backward: input Fourier-space field shape: {u_hat.shape}")
        ffield.p = u_hat
        rfield = self.fft.real_space_field('rfield')
        self.fft.ifft(ffield, rfield)
        print(f"Backward: output real-space field shape: {rfield.p.shape}")
        return rfield.p

def curl(u_cxyz):
    """Computes the curl of a vector field in real space."""
    nb_grid_pts = u_cxyz.shape[:-1]
    print(f"nb_grid_pts: {nb_grid_pts}")
    
    fft = FFT(nb_grid_pts)
    
    # Fourier transform of the input vector field
    u_hat_shape = fft.forward(u_cxyz[..., 0]).shape
    u_hat = np.zeros(u_hat_shape + (3,), dtype=np.complex128)
    print(f"Initial u_hat shape: {u_hat.shape}")
    
    for i in range(3):
        print(f"Transforming component {i}")
        u_component = u_cxyz[..., i]
        print(f"u_component shape: {u_component.shape}")
        u_hat[..., i] = fft.forward(u_component)
        print(f"u_hat[..., {i}] shape after forward transform: {u_hat[..., i].shape}")
    
    # Create wave number vectors based on the Fourier-transformed shape
    kx = np.fft.fftfreq(u_hat_shape[0]) * 2j * np.pi
    ky = np.fft.fftfreq(u_hat_shape[1]) * 2j * np.pi
    kz = np.fft.fftfreq(u_hat_shape[2]) * 2j * np.pi

    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')

    # Compute the curl in Fourier space
    curl_hat = np.zeros_like(u_hat, dtype=np.complex128)
    print(f"Initial curl_hat shape: {curl_hat.shape}")
    
    curl_hat[..., 0] = ky * u_hat[..., 2] - kz * u_hat[..., 1]
    curl_hat[..., 1] = kz * u_hat[..., 0] - kx * u_hat[..., 2]
    curl_hat[..., 2] = kx * u_hat[..., 1] - ky * u_hat[..., 0]

    print(f"curl_hat shape after computation: {curl_hat.shape}")

    # Inverse Fourier transform to get back to real space
    curl_real_shape = fft.backward(curl_hat[..., 0]).shape
    curl_real = np.zeros(curl_real_shape + (3,), dtype=np.float64)
    print(f"Initial curl_real shape: {curl_real.shape}")
    
    for i in range(3):
        print(f"Inverse transforming component {i}")
        curl_real[..., i] = fft.backward(curl_hat[..., i]).real
        print(f"curl_real[..., {i}] shape after backward transform: {curl_real[..., i].shape}")

    print(f"curl_real shape after backward transform: {curl_real.shape}")
    return curl_real

def test_vanishing_curl():
    nb_grid_pts = (32, 32, 2)
    
    # Initialize the FFT class to access the grid points
    fft = FFT(nb_grid_pts)
    
    u_cxyz = np.ones([*nb_grid_pts, 3])
    curl_u_cxyz = curl(u_cxyz)
    np.testing.assert_allclose(curl_u_cxyz, 0, atol=1e-10)
    print("Test passed: Constant field has vanishing curl.")

def generate_vector_field_and_compute_curl():
    nb_grid_pts = (32, 32, 2)
    fft = FFT(nb_grid_pts)# Generate coordinates
    coords = np.array(np.meshgrid(np.linspace(0, 1, nb_grid_pts[0]),
                                  np.linspace(0, 1, nb_grid_pts[1]),
                                  np.linspace(0, 1, nb_grid_pts[2]), indexing='ij'))
    
    norm = np.array([0, 0, 1])
    u_cxyz = np.zeros(coords.shape[1:] + (3,))
    u_cxyz[..., 0] = norm[1] * (coords[2] - 0.5) - norm[2] * (coords[1] - 0.5)
    u_cxyz[..., 1] = norm[2] * (coords[0] - 0.5) - norm[0] * (coords[2] - 0.5)
    u_cxyz[..., 2] = norm[0] * (coords[1] - 0.5) - norm[1] * (coords[0] - 0.5)
    # Compute the curl
    curl_u_cxyz = curl(u_cxyz)
    return coords, u_cxyz, curl_u_cxyz

def plot_vector_field_and_curl(coords, u_cxyz, curl_u_cxyz):
    fig = plt.figure(figsize=(14, 6))
    
    # Plot vector field
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.quiver(coords[0], coords[1], coords[2], u_cxyz[..., 0], u_cxyz[..., 1], u_cxyz[..., 2], 
               length=0.1, normalize=True, color='b', linewidth=0.5)
    ax1.set_title('Vector Field')
    
    # Plot curl of the vector field
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.quiver(coords[0], coords[1], coords[2], curl_u_cxyz[..., 0], curl_u_cxyz[..., 1], curl_u_cxyz[..., 2], 
               length=0.1, normalize=True, color='r', linewidth=0.5)
    ax2.set_title('Curl of the Vector Field')
    
    plt.show()

def print_curl_values(curl_u_cxyz):
    # Print some sample values of the curl
    print("Curl values at various grid points:")
    for i in range(0, curl_u_cxyz.shape[0], max(1, curl_u_cxyz.shape[0] // 5)):
        for j in range(0, curl_u_cxyz.shape[1], max(1, curl_u_cxyz.shape[1] // 5)):
            for k in range(0, curl_u_cxyz.shape[2], max(1, curl_u_cxyz.shape[2] // 2)):
                print(f"Curl at ({i}, {j}, {k}): {curl_u_cxyz[i, j, k]}")

def test_nonvanishing_curl():
    print("Task 4: Test nonvanishing curl")
    nb_grid_pts = (32, 32, 2)
    
    # Initialize the FFT class to access the grid points
    fft = FFT(nb_grid_pts)
    
    # Vector field defined as u_cxyz = np.cross(norm, fft.coords - 0.5, axis=0)
    norm = np.array([0, 0, 1])
    coords = np.array(np.meshgrid(np.linspace(0, 1, nb_grid_pts[0]),
                                  np.linspace(0, 1, nb_grid_pts[1]),
                                  np.linspace(0, 1, nb_grid_pts[2]), indexing='ij'))
    u_cxyz = np.cross(norm, coords - 0.5, axis=0)
    u_cxyz = np.moveaxis(u_cxyz, 0, -1)  # Move the component axis to the last dimension
    
    # Compute the curl
    curl_u_cxyz = curl(u_cxyz)
    
    # Plot the results
    plot_vector_field_and_curl(coords, u_cxyz, curl_u_cxyz)

    # Print the curl values
    print_curl_values(curl_u_cxyz)
    
    # Print the mean curl values
    mean_curl = np.mean(curl_u_cxyz, axis=(0, 1, 2))
    print(f"Mean curl: {mean_curl}")

# Task 3
test_vanishing_curl()

# Task 4
test_nonvanishing_curl()
