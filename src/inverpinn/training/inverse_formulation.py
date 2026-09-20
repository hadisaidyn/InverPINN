"""Revision-7 variable projection and block-coordinate PINN experiments.

Only sparse tensor records enter these functions. No file/truth/metric inputs
exist. Original B_revised training is imported unchanged for J0 by the runner.
"""

import logging
import torch

from inverpinn.models.constrained_concentration import ConstrainedConcentrationMLP
from inverpinn.physics.autodiff import autograd_pde_residual
from inverpinn.physics.profiled_source import ProfiledGaussianSource, profile_Q
from inverpinn.training.development_candidates import loss_factors
from inverpinn.training.forward_pinn import require_finite
from inverpinn.training.losses import combine_losses, log_losses
from inverpinn.training.single_source import SparseTrainingData, TrainingSettings


def activate_block(model, source, block):
    """Explicitly freeze the complementary block; never reset Adam state."""
    if block not in ("joint", "field", "source"):
        raise ValueError("Unknown parameter block.")
    for p in model.parameters():
        p.requires_grad_(block != "source")
        p.grad = None
    for p in source.parameters():
        p.requires_grad_(block != "field")
        p.grad = None


def block_for_step(step, candidate):
    if candidate["algorithm"] == "differentiable_variable_projection":
        return "joint", step+1
    if candidate["algorithm"] != "alternating":
        raise ValueError("Only J1/J2 use this trainer.")
    size = candidate["field_updates"]+candidate["source_updates"]
    if size*candidate["cycles"] != candidate["total_updates"]:
        raise ValueError("Block schedule does not match fixed budget.")
    return ("field" if step % size < candidate["field_updates"] else "source"), step//size+1


class GradientPathway:
    """Read-only pre-update gradients and full-sensor loss at fixed intervals.

    autograd.grad does not populate .grad or consume RNG. Cosine concerns
    field-parameter gradients, not source data gradients (identically zero).
    Source-block field-gradient diagnostics are explicitly missing.
    """

    def __init__(self, data, interval):
        self.interval = interval
        self.points = torch.cat((data.positions.repeat(len(data.times), 1),
            data.times.repeat_interleave(len(data.positions))[:, None]), 1).float()
        self.targets = data.measurements.reshape(-1, 1).float()
        self.rows = []

    def __call__(self, step, losses, weights, model, source, interior):
        if step != 0 and (step+1) % self.interval:
            return
        fields = [p for p in model.parameters() if p.requires_grad]
        sources = [p for p in source.parameters() if p.requires_grad]
        def gradients(loss, params):
            if not params or not loss.requires_grad:
                return [torch.zeros_like(p) for p in params]
            values = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            return [torch.zeros_like(p) if g is None else g for p, g in zip(params, values)]
        row = dict(epoch=step+1, **{f"pre_{k}":v for k,v in source.estimates().items()})
        if fields:
            gd = torch.cat([g.flatten() for g in gradients(losses["data"]*weights["data"], fields)])
            gp = torch.cat([g.flatten() for g in gradients(losses["pde"]*weights["pde"], fields)])
            nd, npde = gd.norm(), gp.norm()
            row.update(field_data_gradient_norm=nd.item(), field_pde_gradient_norm=npde.item(),
                field_total_gradient_norm=(gd+gp).norm().item(),
                data_pde_gradient_cosine=(torch.dot(gd,gp)/(nd*npde)).item() if nd>0 and npde>0 else None)
        else:
            row.update(field_data_gradient_norm=None, field_pde_gradient_norm=None,
                       field_total_gradient_norm=None, data_pde_gradient_cosine=None)
        gs = gradients(losses["pde"]*weights["pde"], sources)
        by_id = {id(p):g.norm().item() for p,g in zip(sources,gs)}
        row.update(raw_x_gradient_norm=by_id.get(id(source.raw_x)), raw_y_gradient_norm=by_id.get(id(source.raw_y)))
        data_source = gradients(losses["data"], sources)
        row["data_source_gradient_norm"] = sum(g.square().sum().item() for g in data_source)**.5
        with torch.no_grad():
            row["full_sensor_mse"] = (model(self.points)-self.targets).square().mean().item()
        self.rows.append(row)


def terminal_profile(model, source, data, points, seed):
    """One declared Q-only fitting operation; independent evaluation stays unused."""
    generator = torch.Generator().manual_seed(seed)
    upper = torch.tensor([1.,1.,data.times[-1].item()])
    r0s, gs = [], []
    for p in (torch.rand((points,3),generator=generator)*upper).split(512):
        p = p.requires_grad_()
        r0s.append(autograd_pde_residual(model(p),p,u=data.u,v=data.v,D=data.D,S=0.).detach())
        gs.append(source.unit(p[:,:1],p[:,1:2]).detach())
    q = profile_Q(torch.cat(r0s),torch.cat(gs))
    source.set_Q(q)
    return float(q)


def fit_profiled(data, settings, sigma, seed, formulation, candidate, profiling,
                 logger=None, on_step=None, on_objective=None):
    """Fixed-budget J1/J2; same sampling order as frozen J0 at each update.

    Field blocks optimize data+PDE with source fixed. Source blocks use the
    same objective but only its PDE term has source gradients; R0 is detached.
    The latter saves field backward work without changing location derivatives.
    Q is analytically optimized, never added to any Adam parameter group.
    """
    if type(data) is not SparseTrainingData or type(settings) is not TrainingSettings:
        raise TypeError("Strict sparse-only records required.")
    if candidate["total_updates"] != settings.steps:
        raise ValueError("Candidate budget differs from declared training budget.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed+1)
    source = ProfiledGaussianSource(settings.initial_x,settings.initial_y,settings.initial_Q,sigma)
    model = ConstrainedConcentrationMLP(data.times[-1].item(),settings.width,settings.depth,formulation["constraint"])
    fields, locations = list(model.parameters()), list(source.parameters())
    if candidate["algorithm"] == "alternating":
        optimizers = {"field":torch.optim.Adam(fields,lr=settings.learning_rate),
                      "source":torch.optim.Adam(locations,lr=settings.learning_rate)}
    else:
        optimizers = {"joint":torch.optim.Adam(fields+locations,lr=settings.learning_rate)}
    weights = {k:settings.optimizer_config(seed)["loss_weights"][k]*v for k,v in loss_factors(formulation).items()}
    points = torch.cat((data.positions.repeat(len(data.times),1),data.times.repeat_interleave(len(data.positions))[:,None]),1).float()
    targets = data.measurements.reshape(-1,1).float()
    upper = torch.tensor([1.,1.,data.times[-1].item()])
    history = []
    logger = logger or logging.getLogger(__name__)
    def sample(n):
        return torch.rand((n,3),generator=generator)*upper
    for step in range(settings.steps):
        block, cycle = block_for_step(step,candidate)
        activate_block(model,source,block)
        ids = torch.randint(len(points),(settings.data_batch,),generator=generator)
        interior = sample(settings.collocation_batch).requires_grad_()
        initial = sample(settings.initial_batch); initial[:,2]=0
        edges=[]
        for axis,value in ((0,0.),(0,1.),(1,0.),(1,1.)):
            edge=sample(settings.boundary_per_edge);edge[:,axis]=value;edges.append(edge)
        boundary=torch.cat(edges)
        optimizer=optimizers[block];optimizer.zero_grad()
        r0=autograd_pde_residual(model(interior),interior,u=data.u,v=data.v,D=data.D,S=0.)
        if block=="source":
            r0=r0.detach()
        g=source.unit(interior[:,:1],interior[:,1:2])
        old_q=source.Q.item()
        q=source.Q if block=="field" else profile_Q(r0,g)
        if block!="field":
            source.set_Q(q)
        losses=dict(data=(model(points[ids])-targets[ids]).square().mean(),pde=(r0-q*g).square().mean(),
            initial=model(initial).square().mean(),boundary=model(boundary).square().mean())
        total=combine_losses(losses,weights)
        require_finite(total,f"{block} step {step} objective")
        if on_objective:
            on_objective(step,losses,weights,model,source,interior)
        total.backward()
        for p in fields+locations:
            if p.grad is not None:
                require_finite(p.grad,f"{block} step {step} gradient")
        record=log_losses(logger,step,losses,weights)
        optimizer.step()
        for p in fields+locations:
            require_finite(p,f"{block} step {step} parameter")
        record.update({f"source/{k}":v for k,v in source.estimates().items()})
        record.update(epoch=step+1,block=block,cycle=cycle,effective_Q_update=source.Q.item()-old_q)
        history.append(record)
        if on_step:
            on_step(record)
    activate_block(model,source,"joint")
    terminal_profile(model,source,data,profiling["terminal_fitting_points"],seed+profiling["terminal_generator_seed_offset"])
    return model,source,{k:v.state_dict() for k,v in optimizers.items()},history
