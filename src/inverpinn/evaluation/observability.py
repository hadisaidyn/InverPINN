"""Linear-amplitude inversion, scaled local sensitivity and transparent geometry.

These functions have no dataset/truth/file imports. Candidate source parameters
belong to a forward operator, not hidden truth. Oracle callers are separately
labeled by the experiment script. All PDE evaluations use the unchanged
PyTorch float64 solver; NumPy/SciPy perform finite-dimensional diagnostics.
"""

import time

import numpy as np
from scipy.optimize import least_squares
import torch

from inverpinn.physics.finite_difference import solve_advection_diffusion
from inverpinn.evaluation.fresh_blind import profile_strength


def forward(theta, positions, times, physics, solver, sigma):
    """Time-major sensor vector for candidate (x,y,Q), with fixed physical maps."""
    if set(physics) != {"u", "v", "D"} or set(solver) != {"nx", "ny", "dt", "steps"}:
        raise ValueError("Forward settings may contain only transport and grid fields.")
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (3,) or not np.isfinite(theta).all() or theta[2] < 0:
        raise ValueError("Candidate must be finite x,y,Q with Q>=0.")
    result = solve_advection_diffusion(**solver, **physics,
        sources=[dict(x_s=theta[0], y_s=theta[1], Q=theta[2], sigma=sigma)],
        output_mode="sensor_only", sensor_positions=torch.as_tensor(positions,dtype=torch.float64),
        measurement_times=np.asarray(times).tolist())
    return result["measurements"].numpy().reshape(-1)


def unit_operator(positions, times, physics, solver, sigma):
    """Small per-call exact-key cache; no interpolation or source-truth access."""
    cache = {}
    def response(location):
        key = tuple(map(float,location))
        if key not in cache:
            cache[key] = forward([*key,1.],positions,times,physics,solver,sigma)
        return cache[key]
    return response


def conditional_Q(unit_readings, observations, initial_Q=1.1, concentration_scale=.88):
    r"""Q>=0 LS: max(0,hᵀy/hᵀh), compared with a numerical scalar optimizer.

    Equal observation weights; scale by a fixed concentration unit for
    optimizer numerics only. It does not change the optimum. Neither true Q
    nor the analytic optimum initializes the numerical least-squares fit.
    """
    h,y = np.asarray(unit_readings,dtype=float).reshape(-1), np.asarray(observations,dtype=float).reshape(-1)
    q,mse = profile_strength(h,y)
    if not np.isfinite(initial_Q) or initial_Q<=0 or not np.isfinite(concentration_scale) or concentration_scale<=0:
        raise ValueError("Positive finite initialization and concentration scale required.")
    start=time.perf_counter()
    fit=least_squares(lambda p:(p[0]*h-y)/concentration_scale,[initial_Q],
        jac=lambda p:h[:,None]/concentration_scale,bounds=(0,np.inf),
        ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=100)
    return dict(Q_analytic=q,Q_numerical=float(fit.x[0]),sensor_mse=mse,
        numerical_success=bool(fit.success),numerical_nfev=int(fit.nfev),
        numerical_seconds=time.perf_counter()-start,
        analytic_numerical_difference=float(fit.x[0]-q))


def amplitude_mismatch(inference, reference):
    """Projection amplitude ratio, RMS ratio, L2 and signed bias; no refit.

    a=<reference,inference>/<reference,reference>. a>1 means too strong
    along the reference direction, but shape error remains separately measured.
    Signed bias is mean(inference-reference); normalization uses reference RMS.
    """
    a,b=np.asarray(inference,dtype=float),np.asarray(reference,dtype=float)
    if a.shape!=b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Matched nonempty finite sensor vectors required.")
    energy=float(np.sum(b*b))
    if energy<=0:
        raise ValueError("Amplitude comparison requires nonzero reference signal.")
    rms=float(np.sqrt(np.mean(b*b)))
    return dict(amplitude_ratio=float(np.sum(a*b)/energy),
        rms_ratio=float(np.sqrt(np.sum(a*a)/energy)),relative_observation_l2=float(np.linalg.norm(a-b)/np.sqrt(energy)),
        signed_observation_bias=float(np.mean(a-b)),normalized_signed_bias=float(np.mean(a-b)/rms))


def scaled_jacobian(theta, response, parameter_scales, concentration_scale, step):
    r"""J_z=(∂y/∂theta)diag(s)/C*: central x/y differences, exact Q column.

    Perturb physical x/y by step*s_j, not by relative true parameter value.
    Gaussian forcing is defined beyond sampling bounds, so a central stencil
    is mathematically valid even at the edge of the source-location prior.
    """
    theta,scales=np.asarray(theta,dtype=float),np.asarray(parameter_scales,dtype=float)
    if theta.shape!=(3,) or scales.shape!=(3,) or not np.isfinite(theta).all() or not np.isfinite(scales).all() or np.any(scales<=0) or step<=0 or concentration_scale<=0:
        raise ValueError("Finite parameters and positive fixed scales/step required.")
    h=np.asarray(response(theta[:2]),dtype=float).reshape(-1)
    columns=[]
    for axis in range(2):
        shift=np.zeros(2);shift[axis]=step*scales[axis]
        derivative=(response(theta[:2]+shift)-response(theta[:2]-shift))/(2*step*scales[axis])
        columns.append(theta[2]*np.asarray(derivative).reshape(-1))
    columns.append(h)
    jac=np.column_stack(columns)*scales[None,:]/concentration_scale
    if not np.isfinite(jac).all():
        raise FloatingPointError("Nonfinite forward sensitivity.")
    return h,jac


def information_metrics(jacobian, parameter_scales):
    """SVD-based local information under unit normalized-reading noise.

    No ridge, signal-dependent scaling, posterior claim or actual noise is
    introduced. Rank-deficient covariance/condition values are explicit None.
    Effective Q information removes location-column components by projection.
    """
    jac=np.asarray(jacobian,dtype=float)
    if jac.ndim!=2 or jac.shape[1]!=3 or not np.isfinite(jac).all():
        raise ValueError("Finite scaled Jacobian with three columns required.")
    _,singular,vt=np.linalg.svd(jac,full_matrices=False)
    if len(singular)!=3:
        raise ValueError("At least three observation rows required.")
    rank=int(np.sum(singular>max(jac.shape)*np.finfo(float).eps*singular[0]))
    norms=np.linalg.norm(jac,axis=0)
    projected=jac[:,2]-jac[:,:2]@np.linalg.lstsq(jac[:,:2],jac[:,2],rcond=None)[0]
    result=dict(rank=rank,sigma_max=float(singular[0]),sigma_mid=float(singular[1]),sigma_min=float(singular[2]),
        jacobian_condition=float(singular[0]/singular[-1]) if rank==3 else None,
        information_eigenvalue_min=float(singular[-1]**2),information_eigenvalue_mid=float(singular[1]**2),
        information_eigenvalue_max=float(singular[0]**2),
        information_condition=float((singular[0]/singular[-1])**2) if rank==3 else None,
        information_logdet=float(2*np.log(singular).sum()) if rank==3 else None,
        x_sensitivity_norm=float(norms[0]),y_sensitivity_norm=float(norms[1]),Q_sensitivity_norm=float(norms[2]),
        Q_effective_information=float(np.sum(projected**2)))
    covariance=(vt.T/(singular**2))@vt if rank==3 else None
    for i,key in enumerate(("x","y","Q")):
        result[f"normalized_{key}_std_proxy"]=float(np.sqrt(covariance[i,i])) if rank==3 else None
        result[f"physical_{key}_std_proxy"]=float(np.sqrt(covariance[i,i])*parameter_scales[i]) if rank==3 else None
    return result


def jacobian_stability(theta,response,scaling,settings):
    """Report all preselected step comparisons; never adapt the chosen step."""
    matrices={step:scaled_jacobian(theta,response,scaling["parameter_scales"],scaling["concentration_scale"],step)[1]
              for step in settings["normalized_steps"]}
    selected=matrices[settings["selected_step"]]
    smallest=np.linalg.svd(selected,compute_uv=False)[-1]
    rows=[]
    for step,jac in matrices.items():
        distance=float(np.linalg.norm(jac-selected)/np.linalg.norm(selected))
        s=np.linalg.svd(jac,compute_uv=False)[-1]
        relative=float(abs(s-smallest)/smallest) if smallest>0 else None
        rows.append(dict(normalized_step=step,relative_J_difference=distance,relative_smin_difference=relative,
            stable=distance<=settings["relative_J_tolerance"] and relative is not None and relative<=settings["relative_smin_tolerance"]))
    return selected,rows


def select_starts(locations,unit_responses,observations,count,separation):
    """Rank a predetermined candidate grid by sparse MSE only; no source truth."""
    coordinates,h,y=np.asarray(locations),np.asarray(unit_responses),np.asarray(observations).reshape(-1)
    if coordinates.shape!=(len(h),2) or h.shape[1:]!=(len(y),) or not np.isfinite(h).all() or not np.isfinite(y).all():
        raise ValueError("Candidate grid and sensor response mapping must match.")
    energy=np.sum(h*h,axis=1)
    if np.any(energy<=0):
        raise ValueError("Candidate unit response has no signal.")
    strengths=np.maximum(0,h@y/energy)
    costs=np.mean((strengths[:,None]*h-y)**2,axis=1)
    chosen=[]
    for i in np.lexsort((np.arange(len(costs)),costs)):
        if all(np.linalg.norm(coordinates[i]-coordinates[j])>=separation for j in chosen):
            chosen.append(int(i))
        if len(chosen)==count:
            return coordinates[chosen].copy(),chosen
    raise ValueError("Predetermined grid cannot provide requested separated starts.")


def classical_inverse(observations,positions,times,physics,solver,sigma,locations,unit_responses,settings,scaling,sensitivity):
    """Joint x,y,Q least squares via analytic amplitude variable projection.

    Source truth/paths/metadata are not arguments. All starts use observations
    only. dr/dx = Q*dh/dx+h*dQ/dx, with
    dQ/dx=(dhᵀy-2Q*hᵀdh)/(hᵀh) on the positive branch. No best-by-truth run
    or claim that numerical convergence certifies a global optimum is made.
    """
    y=np.asarray(observations,dtype=float).reshape(-1)
    starts,indices=select_starts(locations,unit_responses,y,settings["start_count"],settings["start_min_separation"])
    response=unit_operator(positions,times,physics,solver,sigma)
    scale=scaling["concentration_scale"]
    step=sensitivity["selected_step"]
    parameter_scales=np.asarray(scaling["parameter_scales"])
    records=[];begin=time.perf_counter()
    for index,location in zip(indices,starts):
        def residual(point):
            h=response(point);q,_=profile_strength(h,y)
            return (q*h-y)/scale
        def derivative(point):
            h=response(point);q,_=profile_strength(h,y)
            columns=[]
            for axis in range(2):
                shift=np.zeros(2);shift[axis]=step*parameter_scales[axis]
                dh=(response(point+shift)-response(point-shift))/(2*shift[axis])
                dq=float((dh@y-2*q*(h@dh))/(h@h)) if q>0 else 0.
                columns.append((q*dh+h*dq)/scale)
            return np.column_stack(columns)
        run_start=time.perf_counter()
        fit=least_squares(residual,location,jac=derivative,bounds=tuple(settings["bounds"]),
            x_scale=parameter_scales[:2],max_nfev=settings["max_nfev"],
            ftol=settings["ftol"],xtol=settings["xtol"],gtol=settings["gtol"])
        h=response(fit.x);q,mse=profile_strength(h,y)
        records.append(dict(start_index=index,initial_x=float(location[0]),initial_y=float(location[1]),
            x_pred=float(fit.x[0]),y_pred=float(fit.x[1]),Q_pred=q,sensor_mse=mse,
            optimizer_success=bool(fit.success),status=int(fit.status),message=str(fit.message),
            nfev=int(fit.nfev),njev=int(fit.njev),optimality=float(fit.optimality),
            runtime_seconds=time.perf_counter()-run_start))
    # Every start remains in the returned record. Ties resolve by predefined order.
    valid=[i for i,r in enumerate(records) if np.isfinite([r[k] for k in ("x_pred","y_pred","Q_pred","sensor_mse")]).all()]
    if not valid:
        raise FloatingPointError("No finite classical inverse candidate; preserve failure.")
    winner=min(valid,key=lambda i:(records[i]["sensor_mse"],i))
    return records[winner] | dict(selected_start=winner,total_runtime_seconds=time.perf_counter()-begin),records


def geometry(location,positions,physics,sigma,horizon):
    """Downwind projection a=d·w, crosswind b=|d·w_perp|; transparent counts.

    The finite-time corridor is 0<=a/|wind|<=T and
    b<=sqrt(sigma²+2D*a/|wind|). It is a one-width kinematic diagnostic,
    not a sharp plume boundary, optimized sensor score or causal model.
    """
    wind=np.array([physics["u"],physics["v"]],dtype=float)
    speed=float(np.linalg.norm(wind))
    if speed<=0:
        raise ValueError("Directional geometry requires nonzero wind.")
    direction=wind/speed
    delta=np.asarray(positions,dtype=float)-np.asarray(location,dtype=float)
    a=delta@direction;b=np.abs(delta@np.array([-direction[1],direction[0]]))
    distances=np.linalg.norm(delta,axis=1);down=a>=0;tau=a/speed
    reachable=down & (tau<=horizon)
    widths=np.sqrt(sigma**2+2*physics["D"]*np.maximum(tau,0))
    corridor=reachable & (b<=widths)
    x,y=location
    return dict(downwind_sensor_count=int(down.sum()),
        nearest_downwind_distance=float(distances[down].min()) if down.any() else None,
        min_downwind_crosswind_distance=float(b[down].min()) if down.any() else None,
        reachable_downwind_count=int(reachable.sum()),plume_corridor_count=int(corridor.sum()),
        nearest_sensor_distance=float(distances.min()),boundary_distance=float(min(x,y,1-x,1-y)))


def envelope_latent(fields,x,y,times,final_time,active_fraction):
    """Required positive latent C/E and transform gain, excluding E=0 exactly.

    inverse_softplus(g)=g+log(-expm1(-g)) is stable for positive g.
    The gain dC/df=E*(1-exp(-g)) can be tiny simply because C is tiny;
    report both all-interior and predefined active-field statistics.
    """
    yy,xx=np.meshgrid(y,x,indexing="ij")
    records=[];maps=[]
    for field,t in zip(fields,times):
        envelope=16*xx*(1-xx)*yy*(1-yy)*t/final_time
        meaningful=envelope>0
        ratio=np.full_like(field,np.nan,dtype=float)
        ratio[meaningful]=field[meaningful]/envelope[meaningful]
        positive=meaningful & (ratio>0)
        latent=np.full_like(ratio,np.nan)
        latent[positive]=ratio[positive]+np.log(-np.expm1(-ratio[positive]))
        gain=np.full_like(ratio,np.nan)
        gain[meaningful]=envelope[meaningful]*(-np.expm1(-ratio[meaningful]))
        for scope,mask in (("interior",positive),("active",positive & (field>=active_fraction*np.max(field)))):
            row=dict(time=float(t),scope=scope,points=int(mask.sum()),excluded_zero_envelope=int((~meaningful).sum()),
                     zero_latent_limit=int((meaningful & (ratio==0)).sum()))
            for name,values in (("required_positive_latent",ratio),("required_raw_latent",latent),("transform_gain",gain)):
                selected=values[mask]
                for statistic,q in (("min",0.),("q05",.05),("median",.5),("q95",.95),("max",1.)):
                    row[f"{name}_{statistic}"]=float(np.quantile(selected,q)) if selected.size else None
            records.append(row)
        maps.append(dict(envelope=envelope,positive_latent=ratio,raw_latent=latent,transform_gain=gain))
    return records,maps


def network_envelope(model, reference_fields, axis, times, final_time, active_fraction):
    """Post-fit raw output, transform gain and coordinate-gradient diagnostics.

    Gradients are with respect to x,y,t, not optimizer updates. autograd.grad
    never accumulates parameter .grad buffers. Reference fields define only
    an evaluation mask, never training targets or model modification here.
    """
    from inverpinn.models.mlp import ConcentrationMLP
    yy,xx=np.meshgrid(axis,axis,indexing="ij")
    rows=[];maps=[]
    for reference,t in zip(reference_fields,times):
        points=torch.tensor(np.column_stack([xx.ravel(),yy.ravel(),np.full(xx.size,t)]),dtype=torch.float32,requires_grad=True)
        raw=ConcentrationMLP.forward(model,points)
        concentration=model(points)
        raw_gradient=torch.autograd.grad(raw.sum(),points)[0]
        concentration_gradient=torch.autograd.grad(concentration.sum(),points)[0]
        x,y,clock=points.unbind(1)
        envelope=16*x*(1-x)*y*(1-y)*clock/final_time
        values=dict(raw_latent=raw[:,0],positive_latent=torch.nn.functional.softplus(raw[:,0]),
            transform_gain=envelope*torch.sigmoid(raw[:,0]),
            raw_spatial_gradient=torch.linalg.vector_norm(raw_gradient[:,:2],dim=1),
            raw_time_gradient=raw_gradient[:,2].abs(),
            C_spatial_gradient=torch.linalg.vector_norm(concentration_gradient[:,:2],dim=1),
            C_time_gradient=concentration_gradient[:,2].abs())
        arrays={k:v.detach().numpy().reshape(xx.shape) for k,v in values.items()}
        if any(not np.isfinite(a).all() for a in arrays.values()):
            raise FloatingPointError("Nonfinite post-fit envelope/coordinate gradient.")
        interior=(xx>0)&(xx<1)&(yy>0)&(yy<1)
        for scope,mask in (("interior",interior),("active",interior & (reference>=active_fraction*np.max(reference)))):
            row=dict(time=float(t),scope=scope,points=int(mask.sum()))
            for key,array in arrays.items():
                for name,q in (("min",0.),("q05",.05),("median",.5),("q95",.95),("max",1.)):
                    row[f"{key}_{name}"]=float(np.quantile(array[mask],q)) if mask.any() else None
            rows.append(row)
        maps.append(arrays)
    return rows,maps


def failure_evidence(row, thresholds, recovery):
    """Nonexclusive diagnostic flags, with deliberately conservative attribution.

    Map quartiles indicate relative coverage, not a universal identifiability
    threshold. Classical success/PINN failure supports a PINN-specific limit
    under the tested observations. Weak local information plus two failed
    optimizers cannot prove intrinsic nonidentifiability. FD amplitude bias
    is not automatically attributed to a PINN, which does not use that solver.
    """
    classical=bool(row["classical_recovery_success"])
    pinn=bool(row["pinn_recovery_success"])
    weak=row["sigma_min"]<thresholds["sigma_min"]
    weak_Q=row["Q_effective_information"]<thresholds["Q_effective_information"]
    loc=recovery["secondary_localization_max"]
    strength=recovery["secondary_relative_strength_max"]
    location_only=any(row[f"{method}_localization_error"]<=loc and
        row[f"{method}_relative_strength_error"]>strength for method in ("classical","pinn"))
    flags=dict(A=classical and not pinn,B=not classical and not pinn and weak,
        C=location_only and weak_Q,
        D=row["conditional_relative_Q_error"]>strength and
            row["conditional_321_relative_Q_error"]<row["conditional_relative_Q_error"],
        E=classical and pinn)
    if flags["E"]:
        label="E: both recover"; attribution="not a recovery failure"
    elif flags["A"]:
        label="A: classical recovers; PINN fails"; attribution="PINN optimization/model limitation"
    elif not classical and pinn and flags["D"]:
        label="D: classical failure; numerical bias evidence"; attribution="numerical forward-model mismatch"
    elif flags["B"]:
        label="B: both fail; relatively weak information"; attribution="mixed/uncertain"
    elif flags["C"]:
        label="C: location-only recovery; relatively weak Q"; attribution="mixed/uncertain"
    else:
        label="uncertain"; attribution="mixed/uncertain"
    return dict(**{f"flag_{key}":bool(value) for key,value in flags.items()},
        relatively_weak_information=bool(weak),relatively_weak_Q_information=bool(weak_Q),
        evidence_category=label,primary_evidence_attribution=attribution,
        attribution_is_causal_proof=False)
