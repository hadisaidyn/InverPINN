"""InverPINN Explorer: a read-only viewer of the frozen final benchmark.

Run from the repository root: streamlit run demo/app.py
Only saved display arrays and final metric tables are loaded. There is no model,
checkpoint evaluation, forward solver, optimizer or user-provided scientific data.
"""

from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo.evidence import FILTERS, load_bundle, load_fields, filter_scenarios
from demo.plots import FIELD_MODES, field_plot, location_plot, q_plot


@st.cache_data(show_spinner=False)
def evidence():
    return load_bundle()


@st.cache_data(show_spinner=False, max_entries=30)
def fields_for(sid):
    return load_fields(sid)


def select_case(sid):
    st.session_state["failure_filter"] = FILTERS[0]
    st.session_state["scenario"] = sid


def method_metrics(name, row):
    st.markdown(f"#### {name}")
    if row["recovery_success"]:
        st.markdown("**Recovered**")
    else:
        st.markdown("**Did not recover** :red[— retained failure]")
    st.markdown(
        f"""| Frozen metric | Value |
| :-- | --: |
| Localization error | {row['localization_error']:.6f} |
| Relative Q error | {100*row['relative_strength_error']:.3f}% |
| Concentration L2 | {100*row['relative_l2']:.3f}% |
| Q predicted | {row['Q_pred']:.6f} |
"""
    )


def main():
    st.set_page_config(
        page_title="InverPINN Explorer",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.html(
        """<style>
    .stMainBlockContainer {max-width:1280px; padding-top:2rem; padding-bottom:3rem;}
    h1 {font-family:Georgia,serif!important; font-weight:500!important; letter-spacing:-.03em;}
    h2,h3,h4 {letter-spacing:-.015em;}
    [data-testid="stMetricValue"] {font-size:1.7rem; font-variant-numeric:tabular-nums;}
    [data-testid="stMetric"] {border-top:1px solid #d9e0e5; padding-top:.65rem;}
    .stMarkdown table {width:100%; font-variant-numeric:tabular-nums;}
    .stMarkdown td,.stMarkdown th {padding:.28rem .5rem;}
    </style>"""
    )
    try:
        data, manifest = evidence()
    except (ValueError, OSError, KeyError) as exc:
        st.error(f"Evidence verification failed. Viewer stopped: {exc}")
        st.stop()
    st.caption("INVERPINN / FROZEN RESEARCH RESULTS")
    st.title("InverPINN Explorer")
    st.markdown(
        "Interactive viewer for frozen synthetic pollution-source inversion experiments."
    )
    st.info(
        "This demo uses synthetic controlled experiments. It does not identify verified real-world pollution emitters in Almaty. "
        "It is a result viewer, not a live inference engine."
    )
    st.subheader("Final blind benchmark")
    left, middle, right = st.columns(3)
    for column, label, key in [
        (left, "J1 PINN", "J1"),
        (right, "Classical inverse", "classical"),
    ]:
        values = data["summary"][key]
        column.metric(label, f"{values['recovered']} / {values['n']} recovered")
        column.caption(
            f"{100*values['recovery_rate']:.2f}% · median localization {values['metrics']['localization_error']['median']:.6f}"
        )
        column.caption(
            f"Median Q error {100*values['metrics']['relative_strength_error']['median']:.3f}% · field L2 {100*values['metrics']['relative_l2']['median']:.3f}%"
        )
    middle.metric(
        "Preregistered requirement", f"{data['required']} / {len(data['scenarios'])}"
    )
    middle.caption(
        f"{100*data['required']/len(data['scenarios']):.0f}% joint recovery required"
    )
    st.error(
        "J1 did not meet the preregistered reliability threshold. Final confirmation FAILED."
    )
    explorer, global_view, explanation = st.tabs(
        ["Scenario explorer", "Across all 30 scenarios", "How it works"]
    )
    with explorer:
        st.subheader("Final blind benchmark scenarios")
        one, two = st.columns(2)
        one.button(
            "Paper representative / median success · 003",
            on_click=select_case,
            args=(data["paper_cases"]["success"],),
            width="stretch",
        )
        two.button(
            "Paper worst localization failure · 030",
            on_click=select_case,
            args=(data["paper_cases"]["failure"],),
            width="stretch",
        )
        st.caption(
            "Default: lowest scenario ID. Paper success shortcut: median localization rank among J1’s 23 recoveries; not the best case. Worst shortcut: largest localization error across all 30."
        )
        controls, selection = st.columns([1, 2])
        label = controls.selectbox("Explore failures", FILTERS, key="failure_filter")
        scenarios = filter_scenarios(data["scenarios"], label)
        ids = [s["scenario_id"] for s in scenarios]
        if st.session_state.get("scenario") not in ids:
            st.session_state["scenario"] = ids[0]
        sid = selection.selectbox("Scenario", ids, key="scenario")
        scenario = next(s for s in scenarios if s["scenario_id"] == sid)
        st.caption(
            f"{len(ids)} scenarios in this filter. The global benchmark always retains all 30."
        )
        mode_col, time_col = st.columns([2, 1])
        mode = mode_col.selectbox("Concentration view", FIELD_MODES)
        time = time_col.select_slider(
            "Saved time (synthetic units)",
            options=data["times"],
            value=data["times"][-1],
        )
        try:
            fields = fields_for(sid)
        except (ValueError, OSError, KeyError) as exc:
            st.error(f"Stored field unavailable or invalid: {exc}")
            st.stop()
        chart, metrics = st.columns([2.2, 1], gap="large")
        with chart:
            st.plotly_chart(
                field_plot(
                    scenario, fields, data["sensors"], mode, data["times"].index(time)
                ),
                key="field_map",
                theme=None,
                width="stretch",
                config={"displaylogo": False, "scrollZoom": False},
            )
            st.caption(
                "Normalized synthetic domain, not a city map. Star: true source · ×: J1 · diamond: classical · circles: sensors. Hover for values; use the legend to reveal overlapping markers."
            )
            st.caption(
                "81×81 display nodes, sampled from stored 161×161 previews (effective stride 8 of the 641×641 reference). "
                "No spatial interpolation. Concentration views share a scale across methods/times; absolute error has its own scale. Metrics use the original full-resolution evaluation."
            )
        with metrics:
            truth = scenario["truth"]
            st.markdown(
                f"**True source:** ({truth['x_true']:.6f}, {truth['y_true']:.6f})  \n**Q true:** {truth['Q_true']:.6f}"
            )
            method_metrics("J1 PINN", scenario["methods"]["J1"])
            method_metrics("Classical inverse", scenario["methods"]["classical"])
        criterion = data["criterion"]
        st.markdown(
            f"**Joint recovery:** localization error ≤ {criterion['secondary_localization_max']} **AND** relative Q error ≤ {100*criterion['secondary_relative_strength_max']:.0f}%. This is the frozen secondary endpoint."
        )
        if not scenario["reference_target_met"]:
            st.warning(
                f"Retained numerical-reference flag: {100*scenario['reference_error_estimate']:.3f}% Richardson-estimated relative error exceeds the 1% target. This is an estimate, not a certified bound."
            )
        obs = scenario["observability"]
        st.caption(
            f"Sensor RMS: {obs['sensor_rms']:.7f} · smallest scaled Jacobian singular value: {obs['sigma_min']:.7f} · {scenario['information_group']}-information group (frozen pre-fit sensor-RMS split). These are diagnostics, not causal explanations or posterior probabilities."
        )
        with st.expander("Full-precision result, coordinates and provenance"):
            st.json(scenario)
            st.download_button(
                "Download this frozen result",
                json.dumps(scenario, indent=2),
                file_name=f"{sid}.json",
                mime="application/json",
            )
    with global_view:
        st.subheader("True Q vs predicted Q")
        summary = data["summary"]["J1"]
        st.markdown(
            f"**J1: {summary['Q_under']} underestimates / {summary['Q_over']} overestimates.** Source-strength bias persisted."
        )
        st.plotly_chart(
            q_plot(data["scenarios"]),
            key="q_bias",
            theme=None,
            width="stretch",
            config={"displaylogo": False},
        )
        st.caption(
            "Q is Gaussian peak source intensity in synthetic concentration/time units, not integrated emissions."
        )
        st.subheader("True vs predicted source locations")
        show_classical = st.checkbox("Show classical source estimates", value=False)
        st.plotly_chart(
            location_plot(data["scenarios"], show_classical),
            key="source_locations",
            theme=None,
            width="stretch",
            config={"displaylogo": False},
        )
        st.caption(
            "Lines pair each true location with its stored estimate. All 30 final scenarios remain visible; no development scenarios are included."
        )
        st.subheader("Retained J1 failures")
        failed = filter_scenarios(data["scenarios"], "Failed by J1")
        table = []
        for s in failed:
            row = s["methods"]["J1"]
            table.append(
                {
                    "Scenario": s["scenario_id"],
                    "True x": s["truth"]["x_true"],
                    "True y": s["truth"]["y_true"],
                    "J1 x": row["x_pred"],
                    "J1 y": row["y_pred"],
                    "Localization": row["localization_error"],
                    "Q error (%)": 100 * row["relative_strength_error"],
                    "L2 (%)": 100 * row["relative_l2"],
                    "Sensor RMS": s["observability"]["sensor_rms"],
                    "Smallest singular value": s["observability"]["sigma_min"],
                }
            )
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    with explanation:
        st.subheader("What am I looking at?")
        st.markdown(
            "Imagine someone releases pollution somewhere in a city, but you only have a few sensors measuring how much reaches different locations. "
            "The inverse problem asks whether we can work backward to estimate where the pollution came from and how strong the source was. "
            "Here we test that question in a synthetic square, not a real city.\n\n"
            "The colored field shows simulated pollution concentration. The markers show the true source and the locations estimated by the PINN and classical inverse method."
        )
        st.subheader("Reading an inverse problem")
        st.markdown(
            "**Forward physics.** Wind transports pollution and diffusion spreads it. The source adds concentration.\n\n"
            "**Inverse problem.** Sensors measure concentration after transport. The source location and strength must be inferred.\n\n"
            "**PINN.** A PyTorch neural field fits sensor observations and is constrained by the transport PDE. J1 profiles Q analytically from its residual.\n\n"
            "**Classical inverse.** A numerical PDE solver searches source parameters directly using the same sparse observations.\n\n"
            "**Main finding.** The PINN often localized the source accurately but tended to underestimate source strength. Physical validity did not ensure reliable recovery."
        )
        st.latex(
            r"\frac{\partial C}{\partial t}+u\frac{\partial C}{\partial x}+v\frac{\partial C}{\partial y}=D\nabla^2 C+S"
        )
        st.markdown(
            "C = concentration; u,v = wind velocity; D = diffusion coefficient; S = source."
        )
        p = data["physics"]
        st.markdown(
            f"One Gaussian with σ = {data['sigma']}; known wind ({p['u']}, {p['v']}) and D = {p['D']}. Zero initial concentration, zero Dirichlet boundaries and zero added sensor noise. These controlled assumptions do not establish real atmospheric validity."
        )
        st.subheader("Paper and supporting material")
        st.download_button(
            "Read paper · PDF",
            data=(ROOT / "paper/InverPINN_Paper.pdf").read_bytes(),
            file_name="InverPINN_Paper.pdf",
            mime="application/pdf",
        )
        with st.expander("View figure gallery"):
            for path in sorted((ROOT / "paper/generated/figures").glob("0[1-8]_*.png")):
                st.image(
                    str(path), caption=path.stem.replace("_", " "), width="stretch"
                )
        with st.expander("View reproducibility guide"):
            st.markdown(
                "Download the repository guide to keep its original relative-path references. The viewer needs only the committed bundle; rebuilding all display fields requires the preserved research archive."
            )
            st.download_button(
                "Download reproducibility guide",
                data=(ROOT / "docs/reproducibility.md").read_bytes(),
                file_name="reproducibility.md",
                mime="text/markdown",
            )
        st.caption(
            "Repository and deployment instructions are in the README. No explicit reuse license is currently granted."
        )
        with st.expander("Bundle provenance"):
            st.json(
                {
                    "scientific_commit": manifest["scientific_commit"],
                    "display": manifest["display"],
                    "source_inventory_sha256": manifest["source_inventory_sha256"],
                    "input_hashes": data["provenance"]["input_hashes"],
                }
            )
    st.divider()
    st.caption(
        "Research demo using synthetic controlled data. It does not identify verified real-world pollution emitters or provide operational environmental monitoring."
    )


if __name__ == "__main__":
    main()
