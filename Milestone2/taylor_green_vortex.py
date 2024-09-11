import numpy as np
import matplotlib.pyplot as plt
from muFFT import FFT, FourierDerivative

# Set up grid and FFT
N = 64  # Number of grid points in each direction
L = 2 * np.pi  # Domain size
nu = 0.01  # Kinematic viscosity

# Instantiate the FFT object with the PocketFFT engine
fft = FFT([N, N, N], engine='pocketfft')

# Wavenumbers
k = fft.fftfreq
kx, ky, kz = k[0, :, :, :], k[1, :, :, :], k[2, :, :, :]
k_squared = kx**2 + ky**2 + kz**2
k_squared[0, 0, 0] = 1  # Avoid division by zero

def navier_stokes_rhs(t, u_hat):
    """Compute the right-hand side of the Navier-Stokes equations in Fourier space"""
    u = fft.ifft(u_hat)
    
    # Compute nonlinear term in real space
    nonlinear = np.array([
        u[0] * np.gradient(u[0], axis=0) + u[1] * np.gradient(u[0], axis=1) + u[2] * np.gradient(u[0], axis=2),
        u[0] * np.gradient(u[1], axis=0) + u[1] * np.gradient(u[1], axis=1) + u[2] * np.gradient(u[1], axis=2),
        u[0] * np.gradient(u[2], axis=0) + u[1] * np.gradient(u[2], axis=1) + u[2] * np.gradient(u[2], axis=2)
    ])
    
    # Transform nonlinear term to Fourier space
    nonlinear_hat = fft.fft(nonlinear)
    
    # Compute projection to ensure incompressibility
    P_k = 1 - (kx**2 + ky**2 + kz**2) / k_squared
    
    # Compute right-hand side
    rhs = -1j * (
        kx * P_k * nonlinear_hat[0] +
        ky * P_k * nonlinear_hat[1] +
        kz * P_k * nonlinear_hat[2]
    ) - nu * k_squared * u_hat
    
    return rhs

def rk4(f, t, y, dt):
    """Fourth-order Runge-Kutta method"""
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# Set up Taylor-Green vortex initial conditions
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
z = np.linspace(0, L, N, endpoint=False)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

u_init = np.array([
    np.sin(X) * np.cos(Y) * np.cos(Z),
    -np.cos(X) * np.sin(Y) * np.cos(Z),
    np.zeros_like(X)
])
u_hat = fft.fft(u_init)

# Plot initial velocity field
def plot_velocity_field(u, title):
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111, projection='3d')
    norm = np.sqrt(u[0]**2 + u[1]**2 + u[2]**2)
    norm[norm == 0] = 1  # Avoid division by zero
    ax.quiver(X, Y, Z, u[0]/norm, u[1]/norm, u[2]/norm, length=0.1, normalize=True)
    ax.set_xlim([0, L])
    ax.set_ylim([0, L])
    ax.set_zlim([0, L])
    ax.set_title(title)
    plt.show()

plot_velocity_field(u_init, 'Initial Velocity Field')

# Time-stepping parameters
dt = 0.01
T = 1.0
num_steps = int(T / dt)

# Main simulation loop
t = 0
energy_list = []

for step in range(num_steps):
    u_hat += rk4(navier_stokes_rhs, t, u_hat, dt)
    t += dt
    
    energy = np.sum(np.abs(u_hat)**2) / (2 * N**3)
    energy_list.append(energy)
    
    if step % 10 == 0:
        print(f"Step {step}, t = {t:.3f}, Energy = {energy:.6f}")

# Convert final result back to real space
u_final = fft.ifft(u_hat)

# Analytical solution for comparison
u_analytical_hat = u_hat * np.exp(-2 * nu * k_squared * T)
u_analytical = fft.ifft(u_analytical_hat)

# Plot initial and final velocity fields for comparison
def plot_final_fields(u_init, u_final, u_analytical):
    fig = plt.figure(figsize=(18, 6))
    
    norm_init = np.sqrt(u_init[0]**2 + u_init[1]**2 + u_init[2]**2)
    norm_final = np.sqrt(u_final[0]**2 + u_final[1]**2 + u_final[2]**2)
    norm_analytical = np.sqrt(u_analytical[0]**2 + u_analytical[1]**2 + u_analytical[2]**2)
    
    norm_init[norm_init == 0] = 1
    norm_final[norm_final == 0] = 1
    norm_analytical[norm_analytical == 0] = 1

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.quiver(X, Y, Z, u_init[0]/norm_init, u_init[1]/norm_init, u_init[2]/norm_init, length=0.1)
    ax1.set_xlim([0, L])
    ax1.set_ylim([0, L])
    ax1.set_zlim([0, L])
    ax1.set_title('Initial Velocity Field')

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.quiver(X, Y, Z, u_final[0]/norm_final, u_final[1]/norm_final, u_final[2]/norm_final, length=0.1)
    ax2.set_xlim([0, L])
    ax2.set_ylim([0, L])
    ax2.set_zlim([0, L])
    ax2.set_title('Numerical Solution at Final Time')

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.quiver(X, Y, Z, u_analytical[0]/norm_analytical, u_analytical[1]/norm_analytical, u_analytical[2]/norm_analytical, length=0.1)
    ax3.set_xlim([0, L])
    ax3.set_ylim([0, L])
    ax3.set_zlim([0, L])
    ax3.set_title('Analytical Solution at Final Time')

    plt.show()

plot_final_fields(u_init, u_final, u_analytical)

# Plot energy over time
plt.figure()
plt.plot(np.linspace(0, T, num_steps), energy_list)
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Energy of the System Over Time')
plt.show()

print("Simulation complete")