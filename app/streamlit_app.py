"""Dashboard de trazabilidad del framework multiagente."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from qqq_agents.dashboard.data import load_dashboard_artifacts, parse_attribution_cell

ROOT = Path(__file__).resolve().parents[1]


def percentage(value: float) -> str:
    return f"{value:.2%}"


st.set_page_config(
    page_title="Comité multiagente QQQ",
    page_icon="📈",
    layout="wide",
)

st.title("Comité multiagente explicable sobre QQQ")
st.caption("Validación walk-forward 2020-2022. El período final 2023-2024 permanece sin consultar.")

try:
    artifacts = load_dashboard_artifacts(ROOT / "artifacts")
except FileNotFoundError as error:
    st.error(str(error))
    st.code(
        "uv run qqq-agents download\n"
        "uv run qqq-agents prepare\n"
        "uv run qqq-agents walk-forward --with-shap\n"
        "uv run qqq-agents lime-cases",
        language="bash",
    )
    st.stop()

metrics = artifacts.metrics
strategy_metrics = metrics["quantitative_multiagent"]

st.subheader("Resultado de validación")
columns = st.columns(5)
columns[0].metric("Rentabilidad anualizada", percentage(strategy_metrics["annualized_return"]))
columns[1].metric("Sharpe", f"{strategy_metrics['sharpe_ratio']:.2f}")
columns[2].metric("Máximo drawdown", percentage(strategy_metrics["maximum_drawdown"]))
columns[3].metric("Exposición", percentage(strategy_metrics["market_exposure"]))
columns[4].metric("Vetos de riesgo", str(metrics["risk_veto_count"]))

equity_long = (
    artifacts.equity.rename_axis("date")
    .reset_index()
    .melt(id_vars="date", var_name="estrategia", value_name="capital_normalizado")
)
equity_names = {
    "multiagent": "Multiagente",
    "buy_and_hold": "Buy & hold",
    "sma_50_200": "Medias 50/200",
    "single_logistic_agent": "Agente logístico único",
}
equity_long["estrategia"] = (
    equity_long["estrategia"].map(equity_names).fillna(equity_long["estrategia"])
)
equity_figure = px.line(
    equity_long,
    x="date",
    y="capital_normalizado",
    color="estrategia",
    labels={"date": "Fecha", "capital_normalizado": "Capital normalizado", "estrategia": ""},
)
equity_figure.update_layout(hovermode="x unified", legend_orientation="h")
st.plotly_chart(equity_figure, width="stretch")

with st.expander("Comparación numérica", expanded=False):
    comparison = {"Multiagente": strategy_metrics}
    for name, values in metrics["baselines"].items():
        comparison[equity_names.get(name, name)] = values
    comparison_frame = pd.DataFrame(comparison).T.loc[
        :,
        [
            "annualized_return",
            "sharpe_ratio",
            "maximum_drawdown",
            "market_exposure",
            "directional_accuracy",
            "position_changes",
        ],
    ]
    st.dataframe(comparison_frame, width="stretch")

st.divider()
st.subheader("Trazabilidad de una decisión")
available_dates = list(artifacts.decisions.index[::-1])
selected_date = st.selectbox(
    "Fecha de decisión",
    options=available_dates,
    format_func=lambda value: value.strftime("%d/%m/%Y"),
)
decision = artifacts.decisions.loc[selected_date]

summary_columns = st.columns(4)
summary_columns[0].metric("Acción", decision["action"])
summary_columns[1].metric("Puntuación agregada", f"{decision['score_before_veto']:.3f}")
summary_columns[2].metric("Posición resultante", f"{decision['desired_position']:.0f}")
summary_columns[3].metric("Personalidad", str(decision["personality"]).capitalize())

if bool(decision["risk_veto_triggered"]):
    st.warning(f"Veto de riesgo activado: {decision['risk_veto_reason']}")
else:
    st.success("El agente de riesgo no activó el veto.")

agent_ids = ("technical", "momentum", "risk")
signal_frame = pd.DataFrame(
    {
        "agente": ["Técnico", "Momentum", "Riesgo"],
        "señal": [decision[f"{agent_id}_signal"] for agent_id in agent_ids],
        "confianza": [decision[f"{agent_id}_confidence"] for agent_id in agent_ids],
    }
)
signal_long = signal_frame.melt(id_vars="agente", var_name="medida", value_name="valor")
signal_figure = px.bar(
    signal_long,
    x="agente",
    y="valor",
    color="medida",
    barmode="group",
    range_y=[-1, 1],
    labels={"agente": "Agente", "valor": "Valor normalizado", "medida": ""},
)
st.plotly_chart(signal_figure, width="stretch")

st.markdown("#### Factores SHAP más influyentes")
tabs = st.tabs(["Técnico", "Momentum", "Riesgo"])
for tab, agent_id in zip(tabs, agent_ids, strict=True):
    with tab:
        attributions = parse_attribution_cell(decision.get(f"{agent_id}_shap"))
        if not attributions:
            st.info("No hay explicaciones SHAP almacenadas para esta ejecución.")
            continue
        attribution_frame = pd.DataFrame(attributions)
        attribution_frame["direccion"] = attribution_frame["contribution"].apply(
            lambda value: "Favorece clase positiva" if value >= 0 else "Favorece clase negativa"
        )
        attribution_figure = px.bar(
            attribution_frame.sort_values("contribution"),
            x="contribution",
            y="feature",
            color="direccion",
            orientation="h",
            labels={"contribution": "Contribución SHAP", "feature": "Variable", "direccion": ""},
        )
        st.plotly_chart(attribution_figure, width="stretch")
        st.dataframe(
            attribution_frame.loc[:, ["feature", "value", "contribution"]],
            width="stretch",
            hide_index=True,
        )

st.divider()
st.subheader("Casos representativos explicados con LIME")
if artifacts.lime_cases is None:
    st.info("Ejecuta 'qqq-agents lime-cases' para generar esta sección.")
else:
    lime_cases = pd.DataFrame(artifacts.lime_cases["cases"])
    selected_case = st.selectbox(
        "Caso",
        options=list(artifacts.lime_cases["selection"]),
        format_func=lambda value: value.replace("_", " ").capitalize(),
    )
    matching = lime_cases.loc[lime_cases["case_type"] == selected_case]
    st.caption(
        f"Fecha: {matching.iloc[0]['date']} · Acción: {matching.iloc[0]['action']} · "
        f"Veto: {'sí' if matching.iloc[0]['risk_veto_triggered'] else 'no'}"
    )
    for _, case in matching.iterrows():
        with st.expander(f"Agente {case['agent_id']}"):
            st.dataframe(pd.DataFrame(case["attributions"]), width="stretch", hide_index=True)

st.divider()
st.caption(
    "Artefacto académico experimental. No constituye asesoramiento financiero ni está preparado "
    "para ejecutar operaciones reales."
)
