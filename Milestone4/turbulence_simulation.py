import numpy as np
from muFFT import FFT
import matplotlib.pyplot as plt

# Simulation Parameters
GRID_SIZE = 32  # 32x32x32 grid
BOX_LENGTH = 1.0  # Simulation domain size
VISCOSITY = 1/1600  # Kinematic viscosity
TIME_STEP = 0.01  # Simulation time step
TOTAL_SIM_TIME = 5.0  # Total simulation time
INIT_VELOCITY_SCALE = 1.0  # Initial velocity amplitude

# FFT Initialization
grid_dims = (GRID_SIZE, GRID_SIZE, GRID_SIZE)
fft_op = FFT(grid_dims, engine='pocketfft')

# Grid Spacing
spacing = BOX_LENGTH / GRID_SIZE

# Wavevector Computation
wave_vec = (2 * np.pi * fft_op.fftfreq.T / spacing).T
zero_k_mask = (wave_vec.T == np.zeros(3, dtype=int)).T.all(axis=0)
wave_vec_sq = np.sum(wave_vec ** 2, axis=0)

# Initialize Velocity Field Function
def init_velocity_field():
    velocity_field = np.zeros((3,) + fft_op.nb_fourier_grid_pts, dtype=complex)
    rng = np.random.default_rng()
    velocity_field.real = rng.standard_normal(velocity_field.shape)
    velocity_field.imag = rng.standard_normal(velocity_field.shape)
    
    # Apply scaling to initialize velocity field
    scale_factor = np.zeros_like(wave_vec_sq)
    scale_factor[np.logical_not(zero_k_mask)] = INIT_VELOCITY_SCALE * \
        wave_vec_sq[np.logical_not(zero_k_mask)] ** (-5 / 6)
    velocity_field *= scale_factor
    
    # Enforce incompressibility
    k_dot_u = np.sum(wave_vec * velocity_field, axis=0)
    for i in range(3):
        velocity_field[i] -= (wave_vec[i] * k_dot_u) / np.where(wave_vec_sq == 0, 1, wave_vec_sq)
    
    velocity_field[:, zero_k_mask] = 0
    return velocity_field

# Aliasing Correction (2/3 Rule) Application
def enforce_2thirds_rule(field_data):
    mask = np.ones(field_data.shape, dtype=bool)
    for i in range(3):
        mask[:, fft_op.nb_fourier_grid_pts[i]//3:2*fft_op.nb_fourier_grid_pts[i]//3] = False
    return field_data * mask

# Navier-Stokes RHS Computation
def navier_stokes_rhs(velocity_hat, aliasing_correction=False):
    velocity_real = np.array([fft_op.ifft(velocity_hat[i]) for i in range(3)])
    
    if aliasing_correction:
        velocity_real = enforce_2thirds_rule(velocity_real)
    
    nonlinear_term = np.array([
        fft_op.fft(velocity_real[1]*velocity_real[2]),
        fft_op.fft(velocity_real[2]*velocity_real[0]),
        fft_op.fft(velocity_real[0]*velocity_real[1])
    ])
    
    if aliasing_correction:
        nonlinear_term = enforce_2thirds_rule(nonlinear_term)
    
    # Compute Navier-Stokes RHS in Fourier space
    rhs = -1j * np.cross(wave_vec, nonlinear_term, axis=0)
    rhs -= VISCOSITY * wave_vec_sq * velocity_hat
    
    # Enforce incompressibility
    k_dot_rhs = np.sum(wave_vec * rhs, axis=0)
    for i in range(3):
        rhs[i] -= (wave_vec[i] * k_dot_rhs) / np.where(wave_vec_sq == 0, 1, wave_vec_sq)
    
    return rhs

# Fourth-order Runge-Kutta Scheme
def rk4_integrate_step(velocity_hat, time_step, aliasing_correction=False):
    k1 = navier_stokes_rhs(velocity_hat, aliasing_correction)
    k2 = navier_stokes_rhs(velocity_hat + 0.5 * time_step * k1, aliasing_correction)
    k3 = navier_stokes_rhs(velocity_hat + 0.5 * time_step * k2, aliasing_correction)
    k4 = navier_stokes_rhs(velocity_hat + time_step * k3, aliasing_correction)
    return velocity_hat + (time_step/6) * (k1 + 2*k2 + 2*k3 + k4)

# Forcing Function for Low-Wavenumber Modes
def apply_forcing_function(velocity_hat):
    low_k_mask = (np.sqrt(wave_vec_sq) <= 2*np.pi/BOX_LENGTH)
    velocity_hat[:, low_k_mask] *= np.exp(VISCOSITY * wave_vec_sq[low_k_mask] * TIME_STEP)
    return velocity_hat

# Normalize Velocity Field to Maintain Energy
def normalize_energy(velocity_hat):
    total_energy = np.sum(np.abs(velocity_hat)**2)
    return velocity_hat * np.sqrt(1 / total_energy)

# Spectrum Calculation for Energy and Dissipation
def compute_energy_spectrum(velocity_hat):
    energy_spectrum = np.zeros(GRID_SIZE//2)
    dissipation_spectrum = np.zeros(GRID_SIZE//2)
    
    for i in range(GRID_SIZE//2):
        shell = (i <= np.sqrt(wave_vec_sq)) & (np.sqrt(wave_vec_sq) < i+1)
        energy_spectrum[i] = 0.5 * np.sum(np.abs(velocity_hat[:, shell])**2) / (GRID_SIZE**3)
        dissipation_spectrum[i] = 2 * VISCOSITY * np.sum(wave_vec_sq[shell] * np.abs(velocity_hat[:, shell])**2) / (GRID_SIZE**3)
    
    return energy_spectrum, dissipation_spectrum

# Plotting and Saving Results
def save_plot_results(t, velocity_hat_no_correction, velocity_hat_with_correction):
    # Compute spectra
    energy_spectrum_no_correction, dissipation_spectrum_no_correction = compute_energy_spectrum(velocity_hat_no_correction)
    energy_spectrum_with_correction, dissipation_spectrum_with_correction = compute_energy_spectrum(velocity_hat_with_correction)
    
    # Plot the energy and dissipation spectra
    plt.figure(figsize=(12,6))
    plt.loglog(range(1, GRID_SIZE//2+1), energy_spectrum_no_correction, 
               label='Energy (No Correction)', color='purple')
    plt.loglog(range(1, GRID_SIZE//2+1), energy_spectrum_with_correction, 
               label='Energy (With Correction)', color='orange')
    plt.loglog(range(1, GRID_SIZE//2+1), dissipation_spectrum_no_correction, 
               label='Dissipation (No Correction)', color='green')
    plt.loglog(range(1, GRID_SIZE//2+1), dissipation_spectrum_with_correction, 
               label='Dissipation (With Correction)', color='red')
    plt.loglog(range(1, GRID_SIZE//2+1), np.array(range(1, GRID_SIZE//2+1))**(-5/3), '--', 
               label='k^(-5/3)', color='black')
    plt.xlabel('Wavenumber (k)')
    plt.ylabel('Spectrum')
    plt.legend()
    plt.title(f'Energy and Dissipation Spectra at t={t:.2f}')
    plt.savefig(f'spectra_t{t:.2f}.png')
    plt.close()
    
    # Visualize velocity magnitude field
    velocity_real_no_correction = np.array([fft_op.ifft(velocity_hat_no_correction[i]) for i in range(3)])
    velocity_real_with_correction = np.array([fft_op.ifft(velocity_hat_with_correction[i]) for i in range(3)])
    
    plt.figure(figsize=(10,10))
    plt.imshow(np.sqrt(np.sum(velocity_real_no_correction[:,:,:,GRID_SIZE//2]**2, axis=0)), cmap='Blues')
    plt.colorbar()
    plt.title(f'Velocity Magnitude (No Correction) at t={t:.2f}')
    plt.savefig(f'velocity_field_no_correction_t{t:.2f}.png')
    plt.close()
    
    plt.figure(figsize=(10,10))
    plt.imshow(np.sqrt(np.sum(velocity_real_with_correction[:,:,:,GRID_SIZE//2]**2, axis=0)), cmap='Purples')
    plt.colorbar()
    plt.title(f'Velocity Magnitude (With Correction) at t={t:.2f}')
    plt.savefig(f'velocity_field_with_correction_t{t:.2f}.png')
    plt.close()

# Initialize velocity field
vel_hat_no_correction = init_velocity_field()
vel_hat_with_correction = vel_hat_no_correction.copy()

# Simulation Loop
time = 0
plot_timesteps = [0, TOTAL_SIM_TIME/4, TOTAL_SIM_TIME/2, 3*TOTAL_SIM_TIME/4, TOTAL_SIM_TIME]
next_plot_index = 0

# Save initial plots
save_plot_results(time, vel_hat_no_correction, vel_hat_with_correction)
next_plot_index += 1

while time < TOTAL_SIM_TIME:
    vel_hat_no_correction = rk4_integrate_step(vel_hat_no_correction, TIME_STEP, aliasing_correction=False)
    vel_hat_with_correction = rk4_integrate_step(vel_hat_with_correction, TIME_STEP, aliasing_correction=True)
    
    vel_hat_no_correction = apply_forcing_function(vel_hat_no_correction)
    vel_hat_with_correction = apply_forcing_function(vel_hat_with_correction)
    
    vel_hat_no_correction = normalize_energy(vel_hat_no_correction)
    vel_hat_with_correction = normalize_energy(vel_hat_with_correction)
    
    time += TIME_STEP
    
    if next_plot_index < len(plot_timesteps) and time >= plot_timesteps[next_plot_index]:
        save_plot_results(time, vel_hat_no_correction, vel_hat_with_correction)
        next_plot_index += 1
    
    print(f"Simulated Time: {time:.2f} / {TOTAL_SIM_TIME}")

# Final Save
if next_plot_index < len(plot_timesteps):
    save_plot_results(time, vel_hat_no_correction, vel_hat_with_correction)

print("Simulation Complete.")