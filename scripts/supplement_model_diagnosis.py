"""Complete descriptive diagnostics of existing fits, without fitting/selecting.

The completed report and its hash manifest remain immutable. This supplement
has its own protocol, code/environment snapshot, artifact hashes and completion
lock. Refuse any pre-existing supplement or diagnosis_summary.json. Reading
old benchmark inputs is protected by the development-only process audit hook.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from inverpinn.data.development_only import DEVELOPMENT_IDS, development_hashes, install_development_read_guard, sparse_inputs
from inverpinn.data.synthetic_reference import ROOT, sha256, write_json
from inverpinn.evaluation.diagnosis_supplement import pde_terms, statistics, region_masks, source_motion, scaled_gradient_rows
from inverpinn.evaluation.single_source_benchmark import diagnostic_points
from inverpinn.models.mlp import ConcentrationMLP
from inverpinn.models.constrained_concentration import ConstrainedConcentrationMLP
from inverpinn.physics.diagnostic_source import DiagnosticGaussianSource
from inverpinn.physics.nondimensional import CharacteristicScales
from inverpinn.physics.trainable_source import TrainableGaussianSource
from run_model_diagnosis import snapshot


def load_model(path, final_time):
    """Reconstruct only the recipe saved with a terminal development checkpoint."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    recipe = saved["recipe"]
    training, formulation = recipe["training"], recipe["formulation"]
    args = [training[k] for k in ("initial_x", "initial_y", "initial_Q")]+[recipe["sigma"]]
    if formulation == "baseline":
        model = ConcentrationMLP([0., 0., 0.], [1., 1., final_time], training["width"], training["depth"])
        source = TrainableGaussianSource(*args)
    else:
        model = ConstrainedConcentrationMLP(final_time, training["width"], training["depth"], formulation["constraint"])
        source = DiagnosticGaussianSource(*args, parameterization=formulation["Q_parameterization"])
    model.load_state_dict(saved["model_state_dict"])
    source.load_state_dict(saved["source_state_dict"])
    model.eval()
    return model, source


def verify_manifest(root, manifest):
    """Verify only files explicitly listed in a completed local report."""
    for relative, digest in manifest.items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != digest:
            raise ValueError(f"Frozen artifact changed or escaped report: {relative}")


def supplement(settings):
    config = yaml.safe_load((ROOT/settings["diagnosis_config"]).read_text())
    root, benchmark = ROOT/config["output_dir"], ROOT/config["benchmark_root"]
    install_development_read_guard(benchmark)
    subdir = Path(settings["output_subdir"])
    if len(subdir.parts) != 1 or subdir.is_absolute() or subdir.name in (".", ".."):
        raise ValueError("Supplement output must be a single new directory name.")
    output, summary_path = root/subdir, root/"diagnosis_summary.json"
    if output.exists() or summary_path.exists():
        raise FileExistsError("Refusing to overwrite any completed/partial supplement or summary.")
    lock = json.loads((root/"report_complete.json").read_text())
    if sha256(root/"artifact_hashes.json") != lock["artifact_manifest_sha256"]:
        raise ValueError("Completed report manifest changed.")
    manifest = json.loads((root/"artifact_hashes.json").read_text())
    verify_manifest(root, manifest)
    before = development_hashes(benchmark)
    if before != json.loads((root/"development_data_hashes.json").read_text()):
        raise ValueError("Development references changed before supplementary analysis.")
    # Freeze this descriptive policy before any new numerical computations.
    output.mkdir(exist_ok=False)
    (output/"config.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    snapshot(output)
    torch.set_num_threads(1)
    torch.manual_seed(config["optimization_seed"])
    torch.use_deterministic_algorithms(True)
    scales = CharacteristicScales(**settings["comparison_scales"])
    final_time = scales.time
    Q_scale = scales.concentration/scales.time
    term_rows, regions, parameters, gradient_frames = [], [], [], []
    candidates = list(dict.fromkeys(settings["term_candidates"]+settings["parameter_candidates"]))
    allowed = set(pd.read_csv(root/"candidate_summary.csv").candidate)-{"M0"}
    if not set(candidates).issubset(allowed):
        raise ValueError("Only existing fitted candidates may be diagnosed.")
    for candidate in candidates:
        for identifier in DEVELOPMENT_IDS:
            run = root/"runs"/candidate/identifier
            attempt = json.loads((run/"attempt.json").read_text())
            if not attempt["training_terminated"] or not attempt["computational_success"]:
                raise ValueError("Supplement requires terminal successful checkpoints; retain failures in original report.")
            model, source = load_model(run/"model.pt", final_time)
            values = source.estimates()
            record = dict(candidate=candidate, scenario_id=identifier, **values,
                          Q_transform=getattr(source, "parameterization", "softplus"))
            for key, raw, physical in (("x", source.raw_x, source.x_s), ("y", source.raw_y, source.y_s), ("Q", source.raw_Q, source.Q)):
                record[f"raw_{key}_checkpoint"] = raw.detach().item()
                record[f"{key}_jacobian"] = torch.autograd.grad(physical, raw)[0].item()
            record["log10_Q_over_dtype_tiny"] = float(np.log10(values["Q"]/torch.finfo(source.raw_Q.dtype).tiny))
            parameters.append(record)
            if candidate not in settings["term_candidates"]:
                continue
            inputs = sparse_inputs(benchmark, identifier)
            points, _, _ = diagnostic_points(config["diagnostics"], final_time)
            points.requires_grad_()
            terms = {name: value.detach().numpy() for name, value in pde_terms(
                model(points), points, **inputs["physics"]["transport"],
                S=source(points[:, :1], points[:, 1:2])).items()}
            np.savez_compressed(output/f"terms_{candidate}_{identifier}.npz", coordinates=points.detach().numpy(), **terms)
            for name, values_array in terms.items():
                for basis, factor, units in (("physical", 1., "synthetic concentration/time"),
                                              ("dimensionless", scales.time/scales.concentration, "1")):
                    term_rows.append(dict(candidate=candidate, scenario_id=identifier, term=name,
                        basis=basis, units=units, **statistics(values_array*factor)))
            if identifier in config["residual_map_cases"]:
                with np.load(run/"residual_maps.npz", allow_pickle=False) as maps:
                    yy, xx = np.meshgrid(maps["y"], maps["x"], indexing="ij")
                    for i, time in enumerate(maps["times"]):
                        coordinates = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, time)))
                        masks = region_masks(coordinates, [values["x_s"], values["y_s"]],
                            source_radius=settings["source_radius_sigma_multiple"]*config["sigma"],
                            boundary_width=settings["boundary_band_sigma_multiple"]*config["sigma"])
                        for region, mask in masks.items():
                            regions.append(dict(candidate=candidate, scenario_id=identifier, time=time,
                                region=region, **statistics(maps["residual"][i].ravel()[mask])))
            frame = pd.read_csv(run/"gradient_norms.csv")
            frame = scaled_gradient_rows(frame[frame.component == "pde"], scales)
            frame["candidate"], frame["scenario_id"] = candidate, identifier
            gradient_frames.append(frame)
    frames = dict(pde_term_statistics=pd.DataFrame(term_rows), residual_regions=pd.DataFrame(regions),
                  final_source_parameters=pd.DataFrame(parameters), source_gradient_audit=pd.concat(gradient_frames))
    phases = []
    for (candidate, identifier), frame in frames["source_gradient_audit"].groupby(["candidate", "scenario_id"]):
        for phase, (start, stop) in settings["gradient_phases"].items():
            window = frame[frame.epoch.between(start, stop)]
            if window.empty:
                continue
            phases.append(dict(candidate=candidate, scenario_id=identifier, phase=phase, count=len(window),
                location_scaled_norm_median=float(window.location_scaled_norm.median()),
                Q_scaled_abs_median=float(window.Q_scaled_gradient.abs().median()),
                location_to_Q_ratio_median=float(window.location_to_Q_scaled_ratio.median()),
                Q_downward_gradient_fraction=float((window.Q_scaled_gradient > 0).mean()),
                Q_jacobian_min=float(window.Q_jacobian.min()), Q_jacobian_max=float(window.Q_jacobian.max())))
    frames["source_gradient_phases"] = pd.DataFrame(phases)
    motion, network = [], []
    start, middle, stop = settings["motion_steps"]
    for short, long in settings["motion_pairs"]:
        for identifier in DEVELOPMENT_IDS:
            history = pd.read_csv(root/"runs"/long/identifier/"loss_components.csv")
            for a, b in ((start, middle), (middle, stop)):
                motion.append(dict(candidate=long, scenario_id=identifier, **source_motion(history, a, b, Q_scale)))
            a, _ = load_model(root/"runs"/short/identifier/"model.pt", final_time)
            b, _ = load_model(root/"runs"/long/identifier/"model.pt", final_time)
            va = torch.cat([p.detach().flatten().double() for p in a.parameters()])
            vb = torch.cat([p.detach().flatten().double() for p in b.parameters()])
            network.append(dict(short=short, long=long, scenario_id=identifier,
                start=middle, stop=stop, NN_relative_parameter_displacement=float((vb-va).norm()/va.norm())))
    frames["parameter_motion"], frames["network_motion"] = pd.DataFrame(motion), pd.DataFrame(network)
    for name, frame in frames.items():
        frame.to_csv(output/f"{name}.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, candidate in zip(axes, settings["term_candidates"]):
        frame = frames["pde_term_statistics"].query("basis == 'physical'")
        group = frame[frame.candidate == candidate].groupby("term").rmse
        names = ["C_t", "u_C_x", "v_C_y", "D_C_xx", "D_C_yy", "S", "residual"]
        medians = group.median().reindex(names)
        errors = np.vstack((medians-group.min().reindex(names), group.max().reindex(names)-medians))
        axis.bar(names, medians, yerr=errors, capsize=3, color="#35658a")
        axis.set(title=f"{candidate}: median and case range (n=10)", ylabel="Term RMS [synthetic concentration/time]")
        axis.tick_params(axis="x", rotation=45)
    for extension in ("png", "pdf"):
        fig.savefig(output/f"pde_term_magnitudes.{extension}", dpi=config["plot_dpi"])
    plt.close(fig)
    if before != development_hashes(benchmark):
        raise RuntimeError("Reference mutation during diagnostic supplement.")
    verify_manifest(root, manifest)
    decision = json.loads((root/"selection_decision.json").read_text())
    candidate_table = pd.read_csv(root/"candidate_summary.csv")
    baseline = candidate_table.set_index("candidate").loc["B0"]
    revised = candidate_table.set_index("candidate").loc[decision["selected"]]
    comparison = {name: dict(B0=float(baseline[name]), B_revised=float(revised[name]),
        percent_change=100*float(revised[name]/baseline[name]-1) if baseline[name] != 0 else None)
        for name in baseline.index if name.endswith(("_mean", "_median", "_max"))}
    summary = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        scope="10 existing development cases only; descriptive post-selection supplement; no new fits",
        completed_fits=140, adam_updates=960000, raw_rows_including_derived_multistart=150,
        prior_heldout_used=False, new_heldout_generated=False, original_artifacts_unchanged=len(manifest),
        reference_files_unchanged=len(before), selection_unchanged=decision,
        frozen_config=yaml.safe_load((root/"B_revised.yaml").read_text()),
        frozen_config_sha256=sha256(root/"B_revised.yaml"),
        original_artifact_manifest_sha256=sha256(root/"artifact_hashes.json"),
        candidate_summary=candidate_table.to_dict(orient="records"), comparison=comparison,
        supplemental_tables={name:str((output/f"{name}.csv").relative_to(root)) for name in frames},
        evidence=[
            dict(observation="B0 Q is low in all 10 cases", hypothesis="softplus numerical saturation",
                 experiment="signed component gradients; exact final logits; exp-Q control",
                 result="B0 sampled Jacobian 0.1400..0.7093; exp-Q is not a remedy",
                 conclusion="numerical saturation not supported as main cause; attenuation remains"),
            dict(observation="field and Q can compensate", hypothesis="physics penalty shrinks amplitude",
                 experiment="conditional Q and analytic joint amplitude ray (no fitting)",
                 result="terminal B0 shrinkage only 0.15..2.40%; fitted Q close to conditional PDE optimum",
                 conclusion="mechanism exists but cannot alone explain large bias; field-shape/optimization coupling supported"),
            dict(observation="finite-budget endpoints depend on starts and budget", hypothesis="under-convergence/basin dependence",
                 experiment="Q starts 0.5/1.1/1.5; five locations; identical 6000-update prefixes extended to 12000",
                 result="longer training improves B0 primary medians; difficult cases remain sensitive",
                 conclusion="finite-budget effects confirmed; stationary minima and identifiability unresolved"),
            dict(observation="B0 has negative concentrations and soft BC/IC violations", hypothesis="invalid constraints enable better sensor fits",
                 experiment="positive-only, exact-envelope controls, within-candidate correlations",
                 result="hard constraints improve B2 sensor MSE without leakage; B5 nonnegative and exact IC/BC",
                 conclusion="architectural weakness confirmed; leakage as dominant Q-bias cause not established")],
        limitations=["Selected on ten cases, one sensor geometry and optimization seed; no independent generalization claim",
                     "B5 still underestimates Q in all ten cases; 5/10 joint recovery; maximum-case PDE RMS worsens",
                     "B5 source-initialization robustness was not directly tested; B0 controls cannot certify it",
                     "Intermediate logits reconstructed from rounded physical CSV values, not original raw checkpoints",
                     "Residual regions overlap and exclude exact edges/t=0; no continuous-domain error bound",
                     "Supplement defined after selection and cannot be used to retune or claim preselection evidence",
                     "Historical reference target miss for development_004 retained; runtime not a controlled speed benchmark"])
    # Exclusive write plus preflight means the required summary is never silently replaced.
    with summary_path.open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    hashes = {str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output/"artifact_hashes.json", hashes)
    write_json(output/"complete.json", dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        diagnosis_summary_sha256=sha256(summary_path), manifest_sha256=sha256(output/"artifact_hashes.json"),
        original_report_lock_sha256=sha256(root/"report_complete.json"), no_fits_run=True))
    print(json.dumps(dict(selected=decision["selected"], new_tables=list(frames),
                         original_artifacts_verified=len(manifest), references_unchanged=len(before)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model_diagnosis_supplement.yaml")
    args = parser.parse_args()
    supplement(yaml.safe_load((ROOT/args.config).read_text()))
