"""Exploratory inverse PINN fit with irregular observations and dimensional scaling."""

import logging

import numpy as np
import torch

from inverpinn.models.mlp import ConcentrationMLP
from inverpinn.physics.trainable_source import TrainableGaussianMixture
from inverpinn.physics.case_study import scaled_transport_residual
from inverpinn.training.forward_pinn import require_finite
from inverpinn.training.losses import combine_losses, log_losses


def train_case_study(frame, winds, domain, background, config, *, seed, on_step=None):
    """Train only the supplied training rows; caller must never pass held-out PM.

    Fixed K/sigma Gaussian hypotheses, constant-D physics, regional hourly wind.
    Initial and all-edge concentrations share a constant empirical background
    from the first training hour. These are soft priors, NOT measured fields.
    Parameters are optimized for a fixed budget without best-test checkpoint
    selection. Losses use dimensionless concentration and coordinate scales.
    """
    if not frame.split.eq("train").all():
        raise ValueError("Training routine must receive training rows only.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    generator=torch.Generator().manual_seed(seed+1)
    length=domain["length_m"]
    ts,cs=config["time_scale_seconds"],config["concentration_scale_ug_m3"]
    if min(ts,cs)<=0 or config["steps"]<1:
        raise ValueError("Positive scales and training steps are required.")
    upper=domain["duration_seconds"]/ts
    coordinates=np.column_stack([(frame.x-domain["x0"])/length,(frame.y-domain["y0"])/length,frame.time_seconds/ts])
    points=torch.tensor(coordinates,dtype=torch.float32)
    targets=torch.tensor(frame.pm25.to_numpy()[:,None]/cs,dtype=torch.float32)
    model=ConcentrationMLP([0.,0.,0.],[1.,1.,upper],config["width"],config["depth"])
    specs=[dict(x_s=s["x_fraction"],y_s=s["y_fraction"],Q=s["Q_ug_m3_s"]*ts/cs,sigma=s["sigma_m"]/length) for s in config["sources"]]
    source=TrainableGaussianMixture(specs)
    parameters=list(model.parameters())+list(source.parameters())
    optimizer=torch.optim.Adam(parameters,lr=config["learning_rate"])
    wind=torch.tensor(winds,dtype=torch.float32)
    require_finite(points,"case study coordinates")
    require_finite(targets,"case study targets")
    require_finite(wind,"case study hourly winds")
    logger=logging.getLogger("inverpinn.case_study")
    history=[]
    for epoch in range(config["steps"]):
        ids=torch.randint(len(points),(config["data_batch"],),generator=generator)
        interior=(torch.rand((config["collocation_batch"],3),generator=generator)*torch.tensor([1.,1.,upper])).requires_grad_()
        initial=torch.rand((config["initial_batch"],3),generator=generator)
        initial[:,2]=0
        edges=[]
        for axis,value in ((0,0.),(0,1.),(1,0.),(1,1.)):
            edge=torch.rand((config["boundary_per_edge"],3),generator=generator)*torch.tensor([1.,1.,upper])
            edge[:,axis]=value
            edges.append(edge)
        boundary=torch.cat(edges)
        hour=torch.floor(interior[:,2].detach()*ts/3600).long()
        if torch.any(hour<0) or torch.any(hour>=len(wind)):
            raise ValueError("Collocation outside supplied hourly meteorology.")
        optimizer.zero_grad()
        residual=scaled_transport_residual(model(interior),interior,wind_m_s=wind[hour],
            diffusion_m2_s=config["diffusion_m2_s"],source_scaled=source(interior[:,:1],interior[:,1:2]),
            length_m=length,time_scale_seconds=ts)
        losses=dict(data=(model(points[ids])-targets[ids]).square().mean(),pde=residual.square().mean(),
            initial=(model(initial)-background/cs).square().mean(),boundary=(model(boundary)-background/cs).square().mean())
        for name,value in losses.items():
            require_finite(value,f"case study epoch {epoch}: {name}")
        loss=combine_losses(losses,config["loss_weights"])
        loss.backward()
        for parameter in parameters:
            if parameter.grad is not None:
                require_finite(parameter.grad,"case study gradient")
        optimizer.step()
        for parameter in parameters:
            require_finite(parameter,"case study parameter")
        record=log_losses(logger,epoch,losses,config["loss_weights"])
        record.update(epoch=epoch+1,source_estimates=source.estimates())
        history.append(record)
        if on_step:
            on_step(record)
    return model,source,optimizer,history
