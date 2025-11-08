import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.special import erfi

sec_per_yr = 365*24*3600

def w_dansgaard_johnsen(z, H, zk, ws):
    """
    Dansgaard–Johnsen vertical velocity profile, w(z), with z=0 at bed and z=H at surface.

    Parameters
    ----------
    z : float or array_like
        Height(s) above the bed [m], 0 <= z <= H.
    H : float
        Ice thickness [m].
    zk : float
        Kink height [m] (where the profile transitions). Typical choice: zk ≈ 0.7*H.
    ws : float
        Vertical velocity at the surface [m/s], positive upward.
        In steady thickness, ws = a (accumulation rate in m/s).
        If thickness varies, ws = a - dH/dt.

    Returns
    -------
    w : ndarray
        Vertical velocity at z [m/s], positive upward (usually negative within the ice).

    Notes
    -----
    Piecewise definition (z is height above bed):
        For zk <= z <= H:
            w(z) = -ws * (2*z - zk) / (2*H - zk)
        For 0 <= z < zk:
            w(z) = -ws * (z**2) / ((2*H - zk)*zk)
    References
    ----------
    cited in Montelli & Kingslake (2023), The Cryosphere 17, 195–210, Eqs. (3)–(4).
    """
    
    z = np.asarray(z, dtype=float)
    if not (0 < zk <= H):
        raise ValueError("zk must satisfy 0 < zk <= H")
    if np.any((z < 0) | (z > H)):
        raise ValueError("All z must satisfy 0 <= z <= H")

    w = np.empty_like(z, dtype=float)
    mask_upper = z >= zk
    # Upper region: linear in z
    w[mask_upper] = -ws * (2.0*z[mask_upper] - zk) / (2.0*H - zk)
    # Lower region: quadratic in z
    w[~mask_upper] = -ws * (z[~mask_upper]**2) / ((2.0*H - zk)*zk)
    return w

# --- Example ---
# H = 3000.0    # m
# zk = 0.7*H    # kink height
# a = 0.25/100.0/365/24/3600  # 0.25 m/yr accumulation -> m/s
# ws = a        # steady thickness
# z = np.linspace(0, H, 200)
# w = w_dansgaard_johnsen(z, H, zk, ws)

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
    return 3500.0 - 0.02 * (t / sec_per_yr)  # 20 m per kyr lowering

def T_surf_toy(t):
    """Surface temperature [°C]. Glacial cooling example."""
    # Smooth step from -5°C (early) to -15°C (late)
    lgm_year = -20_000  # year of Last Glacial Maximum
    half_change = 5
    T_midpoint = -36
    return T_midpoint - half_change*np.tanh(-((t- lgm_year*sec_per_yr)/(1_000*sec_per_yr) - 1))

def a_rate(t):
    """Accumulation rate [m/s]; allow time variation if desired."""
    return 0.2 / sec_per_yr  # 0.1 m/yr constant

def w_profile(z, t):
    """Vertical velocity [m/s], positive upward: w = a(t) * z / H(t)."""
    return -a_rate(t) * z / max(H(t), 1e-12)

# --- main assembly function ---
def assemble_system(z, T_prev, t, dt, kappa, q_g, k, w_func, Tsurf_func, cap_temp=True):
    """
    Assemble linear system A*T_new = rhs for one CN + implicit-upwind step.
    Optionally caps temperatures at 0°C before building system.

    Parameters
    ----------
    z : ndarray
        Spatial grid [m].
    T_prev : ndarray
        Temperature field at previous step [°C].
    t : float
        Current time [s].
    dt : float
        Time step [s].
    kappa : float
        Thermal diffusivity [m²/s].
    q_g : float
        Geothermal heat flux [W/m²].
    k : float
        Thermal conductivity [W/m/K].
    cap_temp : bool, default=True
        If True, cap T at 0°C before computing advection/diffusion.
    """
    nz = len(z)
    dz = np.diff(z)[0]
    r = kappa * dt / (2.0 * dz**2)

    # --- cap temperatures at melting point ---
    if cap_temp:
        T_prev = np.minimum(T_prev, 0.0)

    # Matrices
    A = np.zeros((nz, nz))
    B = np.zeros((nz, nz))

    # Build interior rows (1..nz-2)
    w = w_func(z, t)  # vertical velocity profile [m/s]
    T_surf = Tsurf_func  # surface temperature function

    for i in range(1, nz-1):
        # Diffusion (Crank–Nicolson)
        A[i, i-1] += -r
        A[i, i  ] +=  1 + 2*r
        A[i, i+1] += -r

        B[i, i-1] +=  r
        B[i, i  ] +=  1 - 2*r
        B[i, i+1] +=  r

        # Advection (fully implicit, upwind)
        wi = w[i]
        if wi >= 0.0:
            A[i, i  ] += dt * wi / dz
            A[i, i-1] += -dt * wi / dz
        else:
            A[i, i+1] +=  dt * wi / dz
            A[i, i  ] += -dt * wi / dz

    # --- Boundary conditions ---
    # Bottom: -k (T[1]-T[0])/dz = q_g  -> T[0] - T[1] = q_g*dz/k
    A[0,0] =  1.0
    A[0,1] = -1.0
    rhs0 = q_g * dz / k

    # Surface: T = T_surf(t+dt)
    A[-1,-1] = 1.0
    rhsN = T_surf(t + dt)

    # --- RHS ---
    rhs = B @ T_prev
    rhs[0]  = rhs0
    rhs[-1] = rhsN

    # --- optional cap after forming RHS (safety) ---
    if cap_temp:
        rhs = np.minimum(rhs, 0.0)

    return A, rhs
