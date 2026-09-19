"""Read-only loss-gradient diagnostics; no random draws or optimizer changes."""

import torch


def component_gradients(losses, weights, model, source):
    """Measure raw-coordinate and physical-coordinate source derivatives.

    Data/IC/BC have *no direct* source-parameter dependence in a joint PINN.
    Their source derivatives are structurally zero, not a broken gradient.
    The field and source communicate indirectly through the PDE during joint
    optimization. For z -> q(z), dL/dq=(dL/dz)/(dq/dz). We report signed
    derivatives as well as norms; positive dL/dQ means gradient descent pushes
    Q DOWN, though Adam's momentum can make an individual update differ.
    Near numerical saturation the physical derivative is undefined here and
    is recorded as None, never replaced by a misleading zero.
    """
    nn_parameters = list(model.parameters())
    raw_parameters = [source.raw_x, source.raw_y, source.raw_Q]
    physical = [source.x_s, source.y_s, source.Q]
    jacobians = [torch.autograd.grad(q, p, retain_graph=True)[0].detach().item()
                 for q, p in zip(physical, raw_parameters)]
    vectors, rows = {}, []
    for name, loss in losses.items():
        gradients = torch.autograd.grad(loss, nn_parameters+raw_parameters,
                                        retain_graph=True, allow_unused=True)
        nn_gradient = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                                 for p, g in zip(nn_parameters, gradients[:len(nn_parameters)])])
        vectors[name] = nn_gradient
        row = dict(component=name, raw_loss=loss.detach().item(), weight=float(weights[name]),
                   nn_norm=nn_gradient.norm().item(), weighted_nn_norm=nn_gradient.norm().item()*weights[name])
        for key, gradient, jacobian in zip(("x_s", "y_s", "Q"), gradients[len(nn_parameters):], jacobians):
            raw = 0. if gradient is None else gradient.detach().item()
            value = raw/jacobian if jacobian != 0 else None
            row.update({f"{key}_raw_gradient": raw, f"{key}_physical_gradient": value,
                        f"{key}_jacobian": jacobian, f"{key}_raw_norm": abs(raw),
                        f"{key}_physical_norm": None if value is None else abs(value),
                        f"{key}_weighted_gradient": None if value is None else weights[name]*value})
        rows.append(row)
    conflicts = {}
    for left in losses:
        for right in losses:
            if left >= right:
                continue
            a, b = vectors[left], vectors[right]
            denominator = (a.norm()*b.norm()).item()
            conflicts[f"cosine_{left}_{right}"] = (torch.dot(a, b).item()/denominator
                                                    if denominator > 0 else None)
    return rows, conflicts


class ObjectiveAudit:
    """Sample diagnostics before updates 1..10, every interval, and the last.

    Uniform collocation counts are measured around the *estimated*, never true,
    source. Counts refer to one batch, not unique points over the entire fit.
    """

    def __init__(self, steps, interval=100):
        self.steps, self.interval = steps, interval
        self.gradients, self.coverage = [], []

    def __call__(self, step, losses, weights, model, source, interior):
        if not (step < 10 or (step+1) % self.interval == 0 or step == self.steps-1):
            return
        rows, conflicts = component_gradients(losses, weights, model, source)
        estimate = source.estimates()
        for row in rows:
            self.gradients.append(dict(epoch=step+1, **estimate, **row, **conflicts))
        with torch.no_grad():
            distance = ((interior[:, 0]-source.x_s)**2+(interior[:, 1]-source.y_s)**2).sqrt()
        self.coverage.append(dict(epoch=step+1, **estimate, count=len(interior),
            within_sigma=int((distance <= source.sigma).sum()),
            within_2sigma=int((distance <= 2*source.sigma).sum()),
            minimum_distance=distance.min().item()))
