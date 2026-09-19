"""Dimensional-to-scaled transport residual for the exploratory case study."""

from inverpinn.physics.autodiff import autograd_pde_residual


def scaled_transport_residual(c, points, *, wind_m_s, diffusion_m2_s, source_scaled,
                              length_m, time_scale_seconds):
    r"""Residual for xi=(x-x0)/L, eta=(y-y0)/L, tau=(t-t0)/T, c=C/C*.

    c_tau + (u*T/L)c_xi + (v*T/L)c_eta
      - (D*T/L²)(c_xixi+c_etaeta) - s = 0,
    where s = S*T/C*, so S = s*C*/T in ug m^-3 s^-1.

    Wind is uniform in space but may vary hourly. Thus advective and conservative
    horizontal transport agree within each hour (spatial divergence is zero).
    D is a fixed effective diffusivity. This is a 2-D concentration-forcing
    model: no deposition, chemistry, topography, vertical mixing or imported
    pollution term. Estimated S can compensate for these omissions; it is NOT
    a mass emission rate or an identified emitter. Scalar diffusion here assumes
    a locally planar UTM metric, not an exact transformed spherical Laplacian.
    """
    if length_m<=0 or time_scale_seconds<=0 or diffusion_m2_s<0:
        raise ValueError("Require positive spatial/time scales and nonnegative diffusion.")
    return autograd_pde_residual(c,points,u=wind_m_s[:,0:1]*time_scale_seconds/length_m,
        v=wind_m_s[:,1:2]*time_scale_seconds/length_m,
        D=diffusion_m2_s*time_scale_seconds/length_m**2,S=source_scaled)
