import numpy as np
from muFFT import FFT
import matplotlib.pyplot as plt

# Simulation parameters
grid_size = 32  # Grid size (32x32x32)
domain_length = 1.0  # Box length
viscosity = 1/1600  # Viscosity
time_step = 0.01  # Time step
final_time = 5.0  # Total simulation time
initial_velocity_amplitude = 1.0  # Amplitude for initial velocity field

# Set up FFT for spectral transforms
grid_points = (grid_size, grid_size, grid_size)
fft_operator = FFT(grid_points, engine='pocketfft')

# Compute wave vectors and grid spacing
dx = domain_length / grid_size
k_vec = (2 * np.pi * fft_operator.fftfreq.T / dx).T
zero_k_mask = (k_vec.T == np.zeros(3, dtype=int)).T.all(axis=0)
k_squared = np.sum(k_vec ** 2, axis=0)

# Function to create initial velocity field in Fourier space
def initialize_velocity_field():
    velocity_field = np.zeros((3,) + fft_operator.nb_fourier_grid_pts, dtype=complex)
    rng = np.random.default_rng()
    velocity_field.real = rng.standard_normal(velocity_field.shape)
    velocity_field.imag = rng.standard_normal(velocity_field.shape)
    
    scale_factor = np.zeros_like(k_squared)
    scale_factor[np.logical_not(zero_k_mask)] = initial_velocity_amplitude * \
        k_squared[np.logical_not(zero_k_mask)] ** (-5 / 6)
    velocity_field *= scale_factor
    
    # Ensure incompressibility
    k_dot_velocity = np.sum(k_vec * velocity_field, axis=0)
    for i in range(3):
        velocity_field[i] -= (k_vec[i] * k_dot_velocity) / np.where(k_squared == 0, 1, k_squared)
    
    velocity_field[:, zero_k_mask] = 0
    return velocity_field

# Function to compute Navier-Stokes right-hand side in Fourier space
def compute_rhs(u_hat):
    u = np.array([fft_operator.ifft(u_hat[i]) for i in range(3)])
    
    # Compute nonlinear term and project in Fourier space
    nonlin_term = np.array([
        fft_operator.fft(u[1] * u[2]),
        fft_operator.fft(u[2] * u[0]),
        fft_operator.fft(u[0] * u[1])
    ])
    
    rhs = -1j * np.cross(k_vec, nonlin_term, axis=0)
    rhs -= viscosity * k_squared * u_hat
    
    # Ensure incompressibility
    k_dot_rhs = np.sum(k_vec * rhs, axis=0)
    for i in range(3):
        rhs[i] -= (k_vec[i] * k_dot_rhs) / np.where(k_squared == 0, 1, k_squared)
    
    return rhs

# Fourth-order Runge-Kutta stepper for time integration
def runge_kutta_step(u_hat, dt):
    k1 = compute_rhs(u_hat)
    k2 = compute_rhs(u_hat + 0.5 * dt * k1)
    k3 = compute_rhs(u_hat + 0.5 * dt * k2)
    k4 = compute_rhs(u_hat + dt * k3)
    return u_hat + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

# Apply forcing function to low wave-number modes
def apply_turbulence_forcing(u_hat):
    low_k_mask = (np.sqrt(k_squared) <= 2*np.pi/domain_length)
    u_hat[:, low_k_mask] *= np.exp(viscosity * k_squared[low_k_mask] * time_step)
    return u_hat

# Normalize the velocity field to maintain energy
def energy_normalization(u_hat):
    total_energy = np.sum(np.abs(u_hat)**2)
    return u_hat * np.sqrt(1 / total_energy)

# Compute energy and dissipation spectra for analysis
def calculate_spectra(u_hat):
    energy_spectrum = np.zeros(grid_size//2)
    dissipation_spectrum = np.zeros(grid_size//2)
    
    for i in range(grid_size//2):
        shell_mask = (i <= np.sqrt(k_squared)) & (np.sqrt(k_squared) < i+1)
        energy_spectrum[i] = 0.5 * np.sum(np.abs(u_hat[:, shell_mask])**2) / (grid_size**3)
        dissipation_spectrum[i] = 2 * viscosity * np.sum(k_squared[shell_mask] * np.abs(u_hat[:, shell_mask])**2) / (grid_size**3)
    
    return energy_spectrum, dissipation_spectrum

# Save velocity field and spectra as plots
def save_simulation_plots(t, u_hat):
    u_real = np.array([fft_operator.ifft(u_hat[i]) for i in range(3)])
    
    # Velocity field visualization (mid-slice)
    plt.figure(figsize=(10,10))
    plt.imshow(np.sqrt(np.sum(u_real[:,:,:,grid_size//2]**2, axis=0)), cmap='inferno')
    plt.colorbar()
    plt.title(f'Velocity Magnitude at t={t:.2f}')
    plt.savefig(f'velocity_field_t{t:.2f}.png')
    plt.close()
    
    # Compute and plot spectra
    energy_spectrum, dissipation_spectrum = calculate_spectra(u_hat)
    
    plt.figure(figsize=(12,6))
    plt.loglog(range(1, grid_size//2+1), energy_spectrum, label='Energy Spectrum')
    plt.loglog(range(1, grid_size//2+1), dissipation_spectrum, label='Dissipation Spectrum')
    plt.loglog(range(1, grid_size//2+1), np.array(range(1, grid_size//2+1))**(-5/3), '--', label='k^(-5/3)')
    plt.xlabel('Wavenumber (k)')
    plt.ylabel('Spectrum')
    plt.legend()
    plt.title(f'Energy and Dissipation Spectra at t={t:.2f}')
    plt.savefig(f'spectra_t{t:.2f}.png')
    plt.close()

# Initialize velocity field
u_hat = initialize_velocity_field()

# Simulation time loop
t = 0
plot_times = [0, final_time/4, final_time/2, 3*final_time/4, final_time]
plot_index = 0

# Save initial state plot
save_simulation_plots(t, u_hat)
plot_index += 1

while t < final_time:
    u_hat = runge_kutta_step(u_hat, time_step)
    u_hat = apply_turbulence_forcing(u_hat)
    u_hat = energy_normalization(u_hat)
    t += time_step
    
    if plot_index < len(plot_times) and t >= plot_times[plot_index]:
        save_simulation_plots(t, u_hat)
        plot_index += 1
    
    print(f"Time: {t:.2f} / {final_time}")

# Ensure final state is saved
if plot_index < len(plot_times):
    save_simulation_plots(t, u_hat)

print("Simulation complete.")
