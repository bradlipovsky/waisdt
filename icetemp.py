import numpy as np
from scipy.special import erf
from scipy.special import erfi

sec_per_yr = 365*24*3600

def w_dansgaard_johnsen(z, H, ws, mb=0.0, p=0.3):
    """
    Dansgaard–Johnsen vertical velocity profile w(z), with basal melt.

    Parameters
    ----------
    z : array_like
        Height above bed [m].
    H : float
        Ice thickness [m].
    ws : float
        Vertical velocity at surface [m/s], positive upward (≈ accumulation rate).
    mb : float, default=0.0
        Basal melt rate [m/s], positive downward.
    p : float, default=0.3
        Fraction of ice thickness for kink (zk/H).

    Returns
    -------
    w : ndarray
        Vertical velocity [m/s], positive upward (negative within ice).
    """
    z = np.asarray(z, float)
    if not (0 < p <= 1):
        raise ValueError("p must satisfy 0 < p <= 1")

    zk = p * H
    w = np.empty_like(z)

    # Base DJ shape (no melt), always downward (negative)
    mask_upper = z >= zk
    w[mask_upper] = -ws * (2.0*z[mask_upper] - zk) / (2.0*H - zk)
    w[~mask_upper] = -ws * (z[~mask_upper]**2) / ((2.0*H - zk)*zk)

    # Add uniform basal melt contribution (downward)
    # Melt means ice leaving the base, so the whole column shifts downward.
    w -= mb

    return w


def load_Tsurf_interpolator(filename="WDC_age_T.txt"):
    """
    Reads a 2-column file (# Age_kyr  T_C) and returns an interpolation function
    T_surf_interp(t) where t is negative seconds before present.
    """
    # Load data (ignore comment lines)
    data = np.loadtxt(filename, comments="#")
    age_kyr, T_C = data[:, 0], data[:, 1]

    # Convert to time in seconds, NEGATIVE before present
    t_s = -age_kyr * 1e3 * 365 * 24 * 3600

    # Ensure t_s is increasing (np.interp requirement)
    sort_idx = np.argsort(t_s)
    t_s = t_s[sort_idx]
    T_C = T_C[sort_idx]

    # Build linear interpolator (constant beyond range)
    def T_surf_interp(t):
        """Interpolated surface temperature at time t [s] (negative before present)."""
        return np.interp(t, t_s, T_C, left=T_C[0], right=T_C[-1])

    return T_surf_interp
def load_accum_interpolator(filename="WDC_accumulation_combined.csv"):
    """
    Reads a 2-column file (# Age_yr  Accum_m_per_yr) and returns an interpolation function
    a_rate_interp(t) where t is negative seconds before present.

    The accumulation rate is returned in m/s (converted from m/yr).
    """
    # Load first two columns (ignore empty and comment lines)
    data = np.genfromtxt(filename, delimiter=",", comments="#", usecols=(0,1))
    
    # Remove rows with NaN
    data = data[~np.isnan(data[:,1])]
    
    # Extract columns
    age_yr, accum_m_per_yr = data[:,0], data[:,1]
    
    # Convert to time in seconds (negative before present)
    t_s = -age_yr * sec_per_yr
    
    # Ensure increasing time for np.interp
    sort_idx = np.argsort(t_s)
    t_s = t_s[sort_idx]
    accum_m_per_yr = accum_m_per_yr[sort_idx]
    
    # Convert m/yr → m/s
    accum_m_per_s = accum_m_per_yr / sec_per_yr

    # Return interpolation function
    def a_rate_interp(t):
        return np.interp(t, t_s, accum_m_per_s)
    
    return a_rate_interp
# --- analytical steady-state temperature profile ---
def steady_temp_profile(z, H, a, kappa, k, q_g, T_s):
    """
    Steady-state glacier temperature profile for w(z) = -(a z / H).
    z = 0 at bed, z = H at surface.
    """
    z = np.asarray(z, float)
    beta = a / (2.0 * kappa * H)
    if np.isclose(beta, 0.0):
        return T_s + (q_g / k) * (H - z)
    rootb = np.sqrt(beta)
    pref = (q_g / k) * np.sqrt(np.pi) / (2.0 * rootb)
    return T_s + pref * (erf(rootb*H) - erf(rootb*z))

# --- time-dependent functions (edit to taste) ---
def H(t):
    """Surface elevation [m]."""
    return 3327.9183 * np.ones_like(t) # constant thickness
    # return 3500.0 - 0.02 * (t / sec_per_yr)  # 20 m per kyr lowering

    # Subtract a smooth step from 3500 m (early) to 2500 m (late)
    # lgm_year = -25_000  # year of Last Glacial Maximum
    # half_change = 250
    # H_midpoint = 3750
    # answer = H_midpoint + half_change*np.tanh(-((t- lgm_year*sec_per_yr)/(1_000*sec_per_yr) - 1))

    return answer

def T_surf_toy(t):
    """Surface temperature [°C]. Glacial cooling example."""
    # Smooth step from -5°C (early) to -15°C (late)
    lgm_year = -20_000  # year of Last Glacial Maximum
    half_change = 5
    T_midpoint = -36
    return T_midpoint - half_change*np.tanh(-((t- lgm_year*sec_per_yr)/(1_000*sec_per_yr) - 1))

def a_rate(t):
    """Accumulation rate [m/s]; allow time variation if desired."""
    if t<-25_000*sec_per_yr:
        return 0.05 / sec_per_yr  # units m/yr
    else:
        return 0.25 / sec_per_yr  # units m/yr
    
    # return 0.25 / sec_per_yr  # units m/yr constant

def w_profile(z, t):
    """Vertical velocity [m/s], positive upward: w = a(t) * z / H(t)."""
    return -a_rate(t) * z / max(H(t), 1e-12)

# --- main assembly function ---
def assemble_system(z, T_prev, t, dt, kappa, q_g, k, w_func, Tsurf_func, cap_temp=True):
    """
    Assemble linear system A*T_new = rhs for one CN + implicit-upwind step.
    Includes basal melt calculation from basal thermal state.
    """
    nz = len(z)
    dz = np.diff(z)[0]
    r = kappa * dt / (2.0 * dz**2)

    # physical constants for melt calculation
    rho_i = 917.0        # kg/m³
    L = 3.34e5           # J/kg

    # --- cap temperatures ---
    if cap_temp:
        T_prev = np.minimum(T_prev, 0.0)

    # --- basal melt calculation ---
    # Bed temp gradient dT/dz ≈ (T[1]-T[0])/dz
    gradT_bed = (T_prev[1] - T_prev[0]) / dz
    q_cond = -k * gradT_bed               # upward conductive heat flux
    mb = 0.0
    if T_prev[0] >= 0.0:  # only melt if temperate base
        mb = max(0.0, (q_g - q_cond) / (rho_i * L))

    # --- matrices ---
    A = np.zeros((nz, nz))
    B = np.zeros((nz, nz))

    # --- vertical velocity profile with melt ---
    w = w_func(z, t, mb=mb)
    T_surf = Tsurf_func

    for i in range(1, nz-1):
        A[i,i-1] += -r
        A[i,i]   +=  1 + 2*r
        A[i,i+1] += -r

        B[i,i-1] +=  r
        B[i,i]   +=  1 - 2*r
        B[i,i+1] +=  r

        wi = w[i]
        if wi >= 0.0:
            A[i,i]   += dt*wi/dz
            A[i,i-1] += -dt*wi/dz
        else:
            A[i,i+1] +=  dt*wi/dz
            A[i,i]   += -dt*wi/dz

    # --- bottom boundary (flux) ---
    A[0,0], A[0,1] = 1, -1
    rhs0 = q_g * dz / k

    # --- surface boundary (Dirichlet) ---
    A[-1,-1] = 1
    rhsN = T_surf(t + dt)

    rhs = B @ T_prev
    rhs[0]  = rhs0
    rhs[-1] = rhsN

    if cap_temp:
        rhs = np.minimum(rhs, 0.0)

    return A, rhs
