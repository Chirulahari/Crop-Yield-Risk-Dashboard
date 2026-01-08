import dash
from dash import dcc, html
import plotly.graph_objects as go
import numpy as np
from scipy.stats import genextreme as gev, genpareto as gpd


def create_dash_app(flask_app, data_store):
    dash_app = dash.Dash(
        server=flask_app,
        url_base_pathname="/dashboard/"
    )

    def empty_fig(title):
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    # ===============================
    # Layout
    # ===============================
    dash_app.layout = html.Div([
        html.H1("Live Crop Yield Risk Dashboard", style={"textAlign": "center"}),

        # 🔹 METRICS CARDS
        html.Div(id="metrics", style={
            "display": "grid",
            "gridTemplateColumns": "repeat(4, 1fr)",
            "gap": "15px",
            "marginBottom": "30px"
        }),

        # 🔹 PLOTS
        dcc.Graph(id="calibration"),
        dcc.Graph(id="cwt"),
        dcc.Graph(id="residuals"),
        dcc.Graph(id="tail"),
        dcc.Graph(id="risk"),
        dcc.Graph(id="ate"),
    ])

    # ===============================
    # Callback
    # ===============================
    @dash_app.callback(
        [
            dash.Output("metrics", "children"),
            dash.Output("calibration", "figure"),
            dash.Output("cwt", "figure"),
            dash.Output("residuals", "figure"),
            dash.Output("tail", "figure"),
            dash.Output("risk", "figure"),
            dash.Output("ate", "figure"),
        ],
        dash.Input("calibration", "id")
    )
    def update_dashboard(_):
        if not data_store:
            return (
                [html.Div("Upload data to see results")],
                *[empty_fig("Upload data first")] * 6
            )

        # ===============================
        # METRICS
        # ===============================
        metric_cards = [
            ("RMSE", data_store["rmse"]),
            ("R²", data_store["r2"]),
            ("PICP", data_store["picp"]),
            ("CRPS", data_store["crps"]),
            ("VaR 99%", data_store["var_99"]),
            ("CVaR 99%", data_store["cvar_99"]),
            ("Sharpness", data_store["sharpness"]),
            ("CWT", data_store["cwt"]),
        ]

        metrics_ui = [
            html.Div([
                html.H4(name),
                html.H2(value)
            ], style={
                "padding": "15px",
                "background": "#f4f6f8",
                "borderRadius": "10px",
                "textAlign": "center",
                "boxShadow": "0px 0px 8px #ccc"
            })
            for name, value in metric_cards
        ]

        # ===============================
        # DATA
        # ===============================
        y = np.array(data_store["y_test"])
        q50 = np.array(data_store["q50"])
        residuals = np.array(data_store["residuals"])

        # ===============================
        # 1️⃣ Calibration
        # ===============================
        calib = go.Figure()
        calib.add_scatter(x=q50, y=y, mode="markers", opacity=0.6)
        calib.add_scatter(
            x=[y.min(), y.max()],
            y=[y.min(), y.max()],
            mode="lines"
        )
        calib.update_layout(
            title="Calibration Plot",
            xaxis_title="Predicted Yield (Q50, MT/ha)",
            yaxis_title="Observed Yield (MT/ha)"
        )

        # ===============================
        # 2️⃣ Coverage vs Width
        # ===============================
        cwt = go.Figure()
        cwt.add_bar(
            x=["PICP", "Mean Interval Width"],
            y=[data_store["picp"], data_store["sharpness"]]
        )
        cwt.update_layout(
            title="Prediction Interval Coverage vs Width",
            yaxis_title="Value"
        )

        # ===============================
        # 3️⃣ Residual Distribution
        # ===============================
        resid = go.Figure()
        resid.add_histogram(x=residuals, nbinsx=30)
        resid.update_layout(
            title="Residual / CRPS Distribution",
            xaxis_title="Residual",
            yaxis_title="Frequency"
        )

        # ===============================
        # 4️⃣ Tail Fit
        # ===============================
        x_vals = np.linspace(residuals.min(), residuals.max(), 200)
        tail = go.Figure()
        tail.add_scatter(
            x=x_vals,
            y=gev.pdf(x_vals, *data_store["gev_params"]),
            name="GEV"
        )
        tail.add_scatter(
            x=x_vals,
            y=gpd.pdf(x_vals - data_store["threshold"], *data_store["pot_params"]),
            name="POT"
        )
        tail.update_layout(
            title="Tail Fit Comparison (GEV vs POT)",
            xaxis_title="Residual",
            yaxis_title="Density"
        )

        # ===============================
        # 5️⃣ VaR / CVaR
        # ===============================
        risk = go.Figure()
        risk.add_bar(
            x=["VaR 99%", "CVaR 99%"],
            y=[data_store["var_99"], data_store["cvar_99"]]
        )
        risk.update_layout(
            title="Risk Metrics",
            yaxis_title="Yield Loss (MT/ha)"
        )

        # ===============================
        # 6️⃣ Region-wise ATE
        # ===============================
        regions = [r["Region"] for r in data_store["region_ate"]]
        ate_vals = [r["ATE"] for r in data_store["region_ate"]]

        ate = go.Figure()
        ate.add_bar(x=regions, y=ate_vals)
        ate.update_layout(
            title="Region-wise Average Treatment Effect",
            xaxis_title="Region",
            yaxis_title="ATE (MT/ha)"
        )

        return metrics_ui, calib, cwt, resid, tail, risk, ate

    return dash_app
