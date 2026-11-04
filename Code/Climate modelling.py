import numpy as np
import math

def mass_flux(velocity, pressure, latitude):

    R= 3390*1000
    g=3.717
    latitude= math.cos(math.radians(latitude))
    factor= (2*math.pi*R*latitude)/g
    integral= (velocity*pressure)
    return factor*integral

print(mass_flux(2, 500*100, 30))

import numpy as np

# --- 1. Constants & Parameters ---
N = 18                  # Number of latitude bands (10-degree increments)
S0 = 589.0              # Solar Constant for Mars (W/m^2)
sigma = 5.67e-8         # Stefan-Boltzmann constant (if using T^4)
C_heat = 1e7            # Heat capacity of Martian surface (J/m^2/K)
dt = 86400 * 5          # Time step (5 Martian days in seconds)

# Budyko Infrared Constants (Mars-adjusted)
A_ir = 203.3            # Radiation at 0°C
B_ir = 2.09             # Increase per degree C
kt = 0.8                # Meridional Transport coefficient (Low for Mars)

# Albedo Constants
alpha_land = 0.25       # Martian dust/rock
alpha_frost = 0.70      # CO2 Frost
T_critical = 145.0      # CO2 Frost point (Kelvin)

# --- 2. Setup Grid ---
# We use sine of latitude (x) for equal-area bands
x = np.linspace(-1, 1, N) 
phi = np.arcsin(x)      # Latitude in radians
T = np.full(N, 210.0)   # Initial guess for Martian temp (Kelvin)

# Solar Distribution using 2nd Legendre Polynomial (P2)
S2 = -0.50
P2 = 0.5 * (3 * x**2 - 1)
S_dist = S0 * (1 + S2 * P2)

def calculate_albedo(temp):
    """Simple temperature-dependent ice-albedo feedback"""
    albedo = np.where(temp < T_critical, alpha_frost, alpha_land)
    return albedo

# --- 3. Simulation Loop ---
for step in range(500):  # Run for 500 time steps
    T_celsius = T - 273.15
    T_global_mean = np.mean(T)
    
    # A. Calculate Albedo
    albedo = calculate_albedo(T)
    
    # B. Absorbed Solar (S/4 replaced by local S_dist/4)
    Q_in = (S_dist / 4) * (1 - albedo)
    
    # C. Emitted Infrared (Budyko Linear Form)
    I_out = A_ir + B_ir * T_celsius
    
    # D. Meridional Heat Transport (F)
    # Heat flows from local band to global mean
    F_transport = kt * (T - T_global_mean)
    
    # E. Energy Balance Equation: dT/dt = (Q_in - I_out - F_transport) / C
    dT = (Q_in - I_out - F_transport) / C_heat
    T += dT * dt

# --- 4. Display Results ---
print(f"{'Lat (deg)':>10} | {'Temp (K)':>10} | {'Albedo':>10}")
print("-" * 35)
for i in range(N):
    lat_deg = np.degrees(phi[i])
    print(f"{lat_deg:10.1f} | {T[i]:10.2f} | {albedo[i]:10.2f}")

print(f"\nGlobal Average Temperature: {np.mean(T):.2f} K")