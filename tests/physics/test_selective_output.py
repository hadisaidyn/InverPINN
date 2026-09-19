"""Storage regression: the same Euler states, not visually similar solutions."""

import pytest
import torch

from inverpinn.data.sensors import sample_sensors
from inverpinn.physics.finite_difference import solve_advection_diffusion


OPTIONS = dict(nx=17, ny=13, dt=0.005, steps=20, u=-0.35, v=0.15, D=0.005,
    sources=[dict(x_s=0.3, y_s=0.7, Q=1.0, sigma=0.08)])
POSITIONS = torch.tensor([[0, 0], [1, 0.6], [0.37, 0.43], [0.3, 0.7]], dtype=torch.float64)


def test_all_modes_match_native_full_history_exactly():
    full = solve_advection_diffusion(**OPTIONS)
    indices = [0, 3, 8, 20]
    times = [i*OPTIONS["dt"] for i in indices]
    selected = solve_advection_diffusion(**OPTIONS, output_mode="selected_times", output_times=times,
        sensor_positions=POSITIONS, measurement_times=times)
    sensors = solve_advection_diffusion(**OPTIONS, output_mode="sensor_only",
        sensor_positions=POSITIONS, measurement_times=times)
    final = solve_advection_diffusion(**OPTIONS, output_mode="final_only")
    legacy_final = solve_advection_diffusion(**OPTIONS, return_history=False)
    _, expected = sample_sensors(full[indices], selected["x"], selected["y"], positions=POSITIONS)
    for actual, reference in [(final, full[-1]), (legacy_final, full[-1]),
                              (selected["fields"], full[indices]), (selected["measurements"], expected),
                              (sensors["measurements"], expected)]:
        assert (actual-reference).abs().max().item() == 0.0
    assert sensors["fields"].shape == (0, OPTIONS["ny"], OPTIONS["nx"])
    assert torch.equal(selected["field_times"], torch.tensor(times, dtype=torch.float64))
    # Distinct snapshots must not alias reused evolution buffers.
    assert not torch.equal(selected["fields"][1], selected["fields"][-1])


def test_off_step_times_use_bracket_interpolation_and_independent_time_axes():
    full = solve_advection_diffusion(**OPTIONS)
    field_times, sensor_times = [0.002, 0.015, 0.099], [0.001, 0.007, 0.01, 0.1]
    result = solve_advection_diffusion(**OPTIONS, output_mode="selected_times", output_times=field_times,
        sensor_positions=POSITIONS, measurement_times=sensor_times)
    def interpolate(times):
        fields = []
        for t in times:
            ratio = t/OPTIONS["dt"]
            left = int(ratio)
            if left == OPTIONS["steps"]:
                fields.append(full[left])
            else:
                alpha = ratio-left
                fields.append((1-alpha)*full[left]+alpha*full[left+1])
        return torch.stack(fields)
    torch.testing.assert_close(result["fields"], interpolate(field_times), atol=1e-15, rtol=1e-13)
    _, readings = sample_sensors(interpolate(sensor_times), result["x"], result["y"], positions=POSITIONS)
    torch.testing.assert_close(result["measurements"], readings, atol=1e-15, rtol=1e-13)
    assert result["measurement_times"].tolist() == sensor_times


def test_sensor_interpolation_exact_for_bilinear_field_interior():
    # With no transport, C=t*(1+2x+3y+xy) at interior nodes. Choose sensors
    # whose four neighbors are interior; zero boundary nodes are not this field.
    options = OPTIONS | dict(u=0, v=0, D=0, sources=[], forcing=lambda x,y,t: 1+2*x+3*y+x*y)
    p = POSITIONS[2:]
    times = [0.017, 0.063]
    result = solve_advection_diffusion(**options, output_mode="sensor_only",
        sensor_positions=p, measurement_times=times)
    x, y = p.T
    expected = torch.tensor(times, dtype=torch.float64)[:, None]*(1+2*x+3*y+x*y)
    torch.testing.assert_close(result["measurements"], expected, atol=1e-15, rtol=1e-13)


@pytest.mark.parametrize("mode", ["final_only", "selected_times", "sensor_only"])
def test_no_full_history_allocation_in_selective_modes(monkeypatch, mode):
    shapes = []
    for name in ("zeros", "empty", "ones", "full"):
        original = getattr(torch, name)
        def guarded(*args, _original=original, **kwargs):
            value = _original(*args, **kwargs)
            shapes.append(tuple(value.shape))
            assert value.shape != (OPTIONS["steps"]+1, OPTIONS["ny"], OPTIONS["nx"])
            return value
        monkeypatch.setattr(torch, name, guarded)
    extra = {"output_mode": mode}
    if mode == "selected_times": extra["output_times"] = [0.025, 0.1]
    if mode != "final_only": extra.update(sensor_positions=POSITIONS, measurement_times=[0, 0.01, 0.1])
    solve_advection_diffusion(**OPTIONS, **extra)
    assert shapes


@pytest.mark.parametrize("times", [[-0.01], [0.101], [float("nan")], [float("inf")],
                                  [0.02, 0.01], [0.01, 0.01], [[0.01]], []])
def test_invalid_physical_times_are_not_rounded_or_silently_reordered(times):
    with pytest.raises(ValueError):
        solve_advection_diffusion(**OPTIONS, output_mode="selected_times", output_times=times)


@pytest.mark.parametrize("options", [dict(output_mode="unknown"),
    dict(output_mode="sensor_only"), dict(output_mode="full_history", output_times=[0.1]),
    dict(output_mode="sensor_only", output_times=[0.1], sensor_positions=POSITIONS, measurement_times=[0.1]),
    dict(output_mode="selected_times", output_times=[0.1], measurement_times=[0.1]),
    dict(output_mode="selected_times", output_times=[0.1], sensor_positions=POSITIONS),
    dict(output_mode="selected_times", output_times=[0.1], return_history=False)])
def test_invalid_output_contracts(options):
    with pytest.raises(ValueError):
        solve_advection_diffusion(**OPTIONS, **options)


def test_zero_steps_and_signed_forcing_preserve_initial_boundary_values():
    result = solve_advection_diffusion(**(OPTIONS | dict(steps=0)), output_mode="selected_times",
        output_times=[0], sensor_positions=POSITIONS, measurement_times=[0])
    assert torch.count_nonzero(result["fields"]) == torch.count_nonzero(result["measurements"]) == 0
    full = solve_advection_diffusion(**OPTIONS, forcing=lambda x,y,t: -torch.ones_like(x)*t)
    selected = solve_advection_diffusion(**OPTIONS, forcing=lambda x,y,t: -torch.ones_like(x)*t,
        output_mode="selected_times", output_times=[0, 0.1])
    torch.testing.assert_close(selected["fields"], full[[0, -1]], atol=0, rtol=0)
    assert torch.count_nonzero(selected["fields"][:, [0, -1]]) == 0
    assert torch.count_nonzero(selected["fields"][:, :, [0, -1]]) == 0
