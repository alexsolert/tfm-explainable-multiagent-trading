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
st.caption(
    "Validación walk-forward 2020-2022 y test final 2023-2024, abierto una sola vez después "
    "de congelar la especificación."
)

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

equity_names = {
    "hybrid_multiagent": "Multiagente híbrido",
    "multiagent": "Multiagente cuantitativo",
    "buy_and_hold": "Buy & hold",
    "sma_50_200": "Medias 50/200",
    "single_logistic_agent": "Agente logístico único",
}

if (
    artifacts.final_quantitative_metrics is not None
    and artifacts.final_quantitative_equity is not None
):
    final_quantitative = artifacts.final_quantitative_metrics["quantitative_multiagent"]
    final_strategy = (
        artifacts.final_hybrid_metrics["hybrid_multiagent"]
        if artifacts.final_hybrid_metrics is not None
        else final_quantitative
    )
    st.subheader("Test final fuera de muestra · 2023-2024")
    final_columns = st.columns(5)
    final_columns[0].metric(
        "Rentabilidad anualizada", percentage(final_strategy["annualized_return"])
    )
    final_columns[1].metric("Sharpe", f"{final_strategy['sharpe_ratio']:.2f}")
    final_columns[2].metric("Máximo drawdown", percentage(final_strategy["maximum_drawdown"]))
    final_columns[3].metric("Exposición", percentage(final_strategy["market_exposure"]))
    final_columns[4].metric(
        "Coste LLM estimado",
        (
            f"${artifacts.final_hybrid_metrics['llm_estimated_cost_usd']:.3f}"
            if artifacts.final_hybrid_metrics is not None
            else "No ejecutado"
        ),
    )

    final_equity = artifacts.final_quantitative_equity.copy()
    if artifacts.final_hybrid_strategy is not None:
        final_equity["hybrid_multiagent"] = artifacts.final_hybrid_strategy["equity"]
    final_equity_long = (
        final_equity.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="estrategia", value_name="capital_normalizado")
    )
    final_equity_long["estrategia"] = (
        final_equity_long["estrategia"].map(equity_names).fillna(final_equity_long["estrategia"])
    )
    final_equity_figure = px.line(
        final_equity_long,
        x="date",
        y="capital_normalizado",
        color="estrategia",
        labels={"date": "Fecha", "capital_normalizado": "Capital normalizado", "estrategia": ""},
    )
    final_equity_figure.update_layout(hovermode="x unified", legend_orientation="h")
    st.plotly_chart(final_equity_figure, width="stretch")

    with st.expander("Comparación final con baselines", expanded=True):
        final_comparison = {"Multiagente cuantitativo": final_quantitative}
        if artifacts.final_hybrid_metrics is not None:
            final_comparison["Multiagente híbrido"] = final_strategy
        for name, values in artifacts.final_quantitative_metrics["baselines"].items():
            final_comparison[equity_names.get(name, name)] = values
        final_comparison_frame = pd.DataFrame(final_comparison).T.loc[
            :,
            [
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "maximum_drawdown",
                "market_exposure",
                "position_changes",
            ],
        ]
        st.dataframe(final_comparison_frame, width="stretch")
        st.caption(
            "El resultado se presenta sin reajuste posterior: el sistema multiagente redujo la "
            "volatilidad frente a buy & hold y medias 50/200, pero obtuvo menor rentabilidad y "
            "Sharpe."
        )

    st.divider()

metrics = artifacts.metrics
strategy_metrics = (
    artifacts.hybrid_metrics["hybrid_multiagent"]
    if artifacts.hybrid_metrics is not None
    else metrics["quantitative_multiagent"]
)
strategy_label = "Híbrido" if artifacts.hybrid_metrics is not None else "Cuantitativo"

st.subheader(f"Resultado de validación · {strategy_label}")
columns = st.columns(5)
columns[0].metric("Rentabilidad anualizada", percentage(strategy_metrics["annualized_return"]))
columns[1].metric("Sharpe", f"{strategy_metrics['sharpe_ratio']:.2f}")
columns[2].metric("Máximo drawdown", percentage(strategy_metrics["maximum_drawdown"]))
columns[3].metric("Exposición", percentage(strategy_metrics["market_exposure"]))
columns[4].metric("Vetos de riesgo", str(metrics["risk_veto_count"]))

equity_long = (
    (
        artifacts.equity.assign(
            hybrid_multiagent=(
                artifacts.hybrid_strategy["equity"]
                if artifacts.hybrid_strategy is not None
                else artifacts.equity["multiagent"]
            )
        )
        if artifacts.hybrid_strategy is not None
        else artifacts.equity
    )
    .rename_axis("date")
    .reset_index()
    .melt(id_vars="date", var_name="estrategia", value_name="capital_normalizado")
)
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
    comparison = {"Multiagente cuantitativo": metrics["quantitative_multiagent"]}
    if artifacts.hybrid_metrics is not None:
        comparison["Multiagente híbrido"] = strategy_metrics
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
hybrid_decision = (
    artifacts.hybrid_decisions.loc[selected_date]
    if artifacts.hybrid_decisions is not None and selected_date in artifacts.hybrid_decisions.index
    else None
)

summary_columns = st.columns(4)
summary_columns[0].metric("Acción", decision["action"])
summary_columns[1].metric("Puntuación agregada", f"{decision['score_before_veto']:.3f}")
summary_columns[2].metric("Posición resultante", f"{decision['desired_position']:.0f}")
summary_columns[3].metric("Personalidad", str(decision["personality"]).capitalize())

if hybrid_decision is not None:
    st.caption(
        f"Comité híbrido completo: {hybrid_decision['action']} · "
        f"puntuación {hybrid_decision['score_before_veto']:.3f}"
    )

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

st.markdown("#### Deliberación híbrida con agentes LLM")
llm_trace = artifacts.llm_traces.get(selected_date.normalize())
if llm_trace is None:
    st.info(
        "No existe una traza LLM para esta fecha. Ejecuta `qqq-agents llm-dry-run --date "
        f"{selected_date.date()}` para generar una simulación gratuita."
    )
else:
    combined = llm_trace["combined_decision"]
    llm_columns = st.columns(4)
    llm_columns[0].metric("Acción híbrida", combined["action"])
    llm_columns[1].metric("Puntuación híbrida", f"{combined['score_before_veto']:.3f}")
    llm_columns[2].metric(
        "Coste incremental estimado",
        f"${llm_trace['incremental_estimated_cost_usd']:.4f}",
    )
    llm_columns[3].metric(
        "Modo",
        "Piloto real" if llm_trace["mode"] == "openai_pilot" else "Simulación",
    )

    contribution_frame = pd.DataFrame(combined["contributions"])
    contribution_figure = px.bar(
        contribution_frame,
        x="agent_id",
        y="contribution",
        color="contribution",
        color_continuous_scale="RdYlGn",
        labels={"agent_id": "Agente", "contribution": "Contribución ponderada"},
    )
    contribution_figure.update_coloraxes(showscale=False)
    st.plotly_chart(contribution_figure, width="stretch")

    role_names = {
        "market_context": "Contexto de mercado",
        "sentiment": "Sentimiento",
        "strategic_validator": "Validación estratégica",
    }
    for result in llm_trace["llm_results"]:
        assessment = result["assessment"]
        with st.expander(role_names.get(result["role"], result["role"])):
            st.write(assessment["justification"])
            st.caption(
                f"Señal: {assessment['signal']:.3f} · Confianza: "
                f"{assessment['confidence']:.3f} · Modelo: {result['model']} · "
                f"Caché: {'sí' if result['cached'] else 'no'}"
            )
            if assessment["limitations"]:
                st.write("Limitaciones: " + "; ".join(assessment["limitations"]))

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
