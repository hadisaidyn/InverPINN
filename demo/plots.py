"""Scientific display of saved coordinates and fields; no forward/inverse solve."""

import numpy as np
import plotly.graph_objects as go

INK = "#1F3443"
J1 = "#16688A"
CLASSICAL = "#B87520"
FIELD_MODES = (
    "Reference concentration",
    "J1 predicted concentration",
    "J1 absolute error",
    "Classical reconstructed concentration",
)


def layout(fig, square=False):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=13, color=INK),
        margin=dict(l=35, r=30, t=58, b=42),
        height=530,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0, font=dict(size=12)),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
    )
    fig.update_xaxes(title="x", range=[0, 1], showgrid=False, zeroline=False)
    fig.update_yaxes(title="y", range=[0, 1], showgrid=False, zeroline=False)
    if square:
        # Preserve [0,1] on both axes: shrink plot domains, not coordinate ranges.
        fig.update_xaxes(constrain="domain")
        fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")
    return fig


def source_trace(name, x, y, q, color, symbol, size=13, ids=None):
    return go.Scatter(
        x=x,
        y=y,
        name=name,
        mode="markers",
        customdata=np.column_stack([q, ids if ids is not None else [""] * len(x)]),
        marker=dict(
            color=color,
            symbol=symbol,
            size=size,
            line=dict(color=INK if name == "True source" else "white", width=1.5),
        ),
        hovertemplate=name
        + "<br>x=%{x:.6f}<br>y=%{y:.6f}<br>Q=%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
    )


def field_plot(scenario, fields, sensors, mode, time_index):
    """All concentration modes share one scale across methods and saved times.

    Absolute error is a display-only subtraction of the two stored previews,
    not a new prediction and never the source of benchmark error metrics.
    """
    if mode not in FIELD_MODES:
        raise ValueError("Unknown field mode")
    if not 0 <= time_index < len(fields["times"]):
        raise ValueError("Unknown saved time")
    key = {
        FIELD_MODES[0]: "reference",
        FIELD_MODES[1]: "J1",
        FIELD_MODES[3]: "classical",
    }.get(mode)
    error = mode == FIELD_MODES[2]
    values = (
        np.abs(fields["J1"].astype(float) - fields["reference"].astype(float))
        if error
        else fields[key]
    )
    limit = (
        float(np.max(values))
        if error
        else max(float(np.max(fields[k])) for k in ("reference", "J1", "classical"))
    )
    fig = go.Figure(
        go.Heatmap(
            x=fields["x"],
            y=fields["y"],
            z=values[time_index],
            zmin=0,
            zmax=limit,
            colorscale="Inferno" if error else "Cividis",
            zsmooth=False,
            colorbar=dict(title="|ΔC|" if error else "C", thickness=12, len=0.8),
            hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>"
            + ("|ΔC|" if error else "C")
            + "=%{z:.7f}<extra></extra>",
        )
    )
    sensors = np.asarray(sensors)
    fig.add_trace(
        go.Scatter(
            x=sensors[:, 0],
            y=sensors[:, 1],
            name="Sensors",
            mode="markers",
            marker=dict(
                size=6, symbol="circle", color="white", line=dict(width=1, color=INK)
            ),
            hovertemplate="Sensor<br>x=%{x:.6f}<br>y=%{y:.6f}<extra></extra>",
        )
    )
    truth = scenario["truth"]
    # Each estimate trace reads its own stored output, not truth-based coordinates.
    for method, color, symbol in [("classical", CLASSICAL, "diamond"), ("J1", J1, "x")]:
        row = scenario["methods"][method]
        fig.add_trace(
            source_trace(
                "Classical estimate" if method == "classical" else "J1 estimate",
                [row["x_pred"]],
                [row["y_pred"]],
                [row["Q_pred"]],
                color,
                symbol,
                14,
            )
        )
    fig.add_trace(
        source_trace(
            "True source",
            [truth["x_true"]],
            [truth["y_true"]],
            [truth["Q_true"]],
            "white",
            "star",
            12,
        )
    )
    return layout(fig, square=True)


def q_plot(scenarios):
    fig = go.Figure()
    truth = [s["truth"]["Q_true"] for s in scenarios]
    estimates = [
        s["methods"][m]["Q_pred"] for m in ("J1", "classical") for s in scenarios
    ]
    low = min(truth + estimates)
    high = max(truth + estimates)
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="Identity: predicted = true",
            line=dict(color="#637482", width=1, dash="dash"),
            hoverinfo="skip",
        )
    )
    for method, color, symbol in [
        ("J1", J1, "x"),
        ("classical", CLASSICAL, "diamond-open"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=truth,
                y=[s["methods"][method]["Q_pred"] for s in scenarios],
                mode="markers",
                name=method,
                text=[s["scenario_id"] for s in scenarios],
                marker=dict(size=10, color=color, symbol=symbol),
                hovertemplate="%{text}<br>True Q=%{x:.7f}<br>Predicted Q=%{y:.7f}<extra>%{fullData.name}</extra>",
            )
        )
    layout(fig)
    fig.update_xaxes(
        title="True Q (synthetic C/time)", range=None, autorange=True, showgrid=True
    )
    fig.update_yaxes(
        title="Predicted Q (synthetic C/time)",
        range=None,
        autorange=True,
        showgrid=True,
    )
    return fig


def location_plot(scenarios, include_classical=False):
    fig = go.Figure()
    for method, color, symbol in [("J1", J1, "x")] + (
        [("classical", CLASSICAL, "diamond-open")] if include_classical else []
    ):
        xs = []
        ys = []
        for s in scenarios:
            xs.extend([s["truth"]["x_true"], s["methods"][method]["x_pred"], None])
            ys.extend([s["truth"]["y_true"], s["methods"][method]["y_pred"], None])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color, width=0.8),
                opacity=0.5,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            source_trace(
                "J1 estimate" if method == "J1" else "Classical estimate",
                [s["methods"][method]["x_pred"] for s in scenarios],
                [s["methods"][method]["y_pred"] for s in scenarios],
                [s["methods"][method]["Q_pred"] for s in scenarios],
                color,
                symbol,
                10,
                [s["scenario_id"] for s in scenarios],
            )
        )
    fig.add_trace(
        source_trace(
            "True source",
            [s["truth"]["x_true"] for s in scenarios],
            [s["truth"]["y_true"] for s in scenarios],
            [s["truth"]["Q_true"] for s in scenarios],
            "white",
            "star",
            11,
            [s["scenario_id"] for s in scenarios],
        )
    )
    return layout(fig, square=True)
