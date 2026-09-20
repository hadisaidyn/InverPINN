"""Hybrid PINN-location plus observation-conditioned forward amplitude.

Not a pure PINN: location is fixed by a PINN, whereas Q and concentration
come from the independent numerical forward operator. No source truth input.
"""

import numpy as np
import torch

from inverpinn.evaluation.observability import conditional_Q, forward
from inverpinn.physics.finite_difference import solve_advection_diffusion
from inverpinn.training.single_source import SparseTrainingData


def observation_profile(data, location, solver, sigma):
    """Fit scalar Q to observed readings using a unit-source numerical solve."""
    if type(data) is not SparseTrainingData or set(location)!={"x_s","y_s"}:
        raise TypeError("Sparse data and x/y-only estimate required; no truth/extra parameters.")
    if not all(np.isfinite(v) and 0<=v<=1 for v in location.values()):
        raise ValueError("Finite bounded location required.")
    h=forward([location["x_s"],location["y_s"],1.],data.positions.numpy(),data.times.numpy(),
        {k:getattr(data,k) for k in ("u","v","D")},solver,sigma)
    result=conditional_Q(h,data.measurements.numpy().reshape(-1))
    q=max(np.finfo(np.float64).tiny,result["Q_analytic"])
    return location | dict(Q=q), result


def numerical_field(data, estimate, solver, sigma, times):
    """Save snapshots plus adjacent steps for a separate centered PDE check.

    Evaluate C_t by adjacent-step difference, spatial derivatives at their
    midpoint, centered in x/y. This is NOT the upwind Euler update residual,
    and NOT identical to off-grid neural autograd. It diagnoses discretization
    at native interior nodes, excluding edges where PDE is not imposed.
    """
    dt=solver["dt"]
    schedule=sorted({0., *times, *(round(t-dt,12) for t in times)})
    solution=solve_advection_diffusion(**solver,**{k:getattr(data,k) for k in ("u","v","D")},
        sources=[estimate | dict(sigma=sigma)],output_mode="selected_times",output_times=schedule)
    allfields=solution["fields"].numpy()
    lookup={round(float(t),12):c for t,c in zip(solution["field_times"],allfields)}
    x,y=solution["x"].numpy(),solution["y"].numpy()
    dx,dy=x[1]-x[0],y[1]-y[0]
    yy,xx=np.meshgrid(y[1:-1],x[1:-1],indexing="ij")
    source=estimate["Q"]*np.exp(-((xx-estimate["x_s"])**2+(yy-estimate["y_s"])**2)/(2*sigma**2))
    residuals=[]
    for t in times:
        current,previous=lookup[round(t,12)],lookup[round(t-dt,12)]
        mid=(current+previous)/2
        ct=(current[1:-1,1:-1]-previous[1:-1,1:-1])/dt
        cx=(mid[1:-1,2:]-mid[1:-1,:-2])/(2*dx)
        cy=(mid[2:,1:-1]-mid[:-2,1:-1])/(2*dy)
        lap=(mid[1:-1,2:]-2*mid[1:-1,1:-1]+mid[1:-1,:-2])/dx**2+(mid[2:,1:-1]-2*mid[1:-1,1:-1]+mid[:-2,1:-1])/dy**2
        residuals.append(ct+data.u*cx+data.v*cy-data.D*lap-source)
    r=np.stack(residuals)
    bc=np.concatenate([allfields[:,0,:].ravel(),allfields[:,-1,:].ravel(),allfields[:,:,0].ravel(),allfields[:,:,-1].ravel()])
    ic=lookup[0.]
    diagnostics=dict(pde_rmse=float(np.sqrt(np.mean(r*r))),pde_mae=float(np.mean(abs(r))),pde_maxabs=float(abs(r).max()),
        pde_diagnostic_method="native_centered_FD_not_neural_autograd",bc_rmse=float(np.sqrt(np.mean(bc*bc))),bc_maxabs=float(abs(bc).max()),
        ic_rmse=float(np.sqrt(np.mean(ic*ic))),ic_maxabs=float(abs(ic).max()))
    return np.stack([lookup[round(t,12)] for t in times]),x,y,diagnostics
