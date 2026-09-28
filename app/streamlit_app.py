"""Aplicación pública de demostración del framework multiagente."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from qqq_agents.config import load_config
from qqq_agents.dashboard.data import DashboardArtifacts, load_dashboard_artifacts

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ARTIFACTS = ROOT / "artifacts"
DEMO_ARTIFACTS = ROOT / "demo_data"
DASHBOARD_SCHEMA_VERSION = "v8-final-1"

COLORS = {
    "Multiagente V2": "#0f766e",
    "Multiagente V3": "#0d9488",
    "Multiagente V4 diario": "#0369a1",
    "Multiagente híbrido": "#14b8a6",
    "Multiagente cuantitativo": "#38bdf8",
    "Buy & hold": "#f59e0b",
    "Medias 50/200": "#a78bfa",
    "Agente logístico único": "#fb7185",
    "Objetivo de volatilidad": "#64748b",
}
EQUITY_NAMES = {
    "v2_multiagent": "Multiagente V2",
    "v3_multiagent": "Multiagente V3",
    "v4_multiagent": "Multiagente V4 diario",
    "hybrid_multiagent": "Multiagente híbrido",
    "multiagent": "Multiagente cuantitativo",
    "buy_and_hold": "Buy & hold",
    "sma_50_200": "Medias 50/200",
    "single_logistic_agent": "Agente logístico único",
    "trailing_volatility_target": "Objetivo de volatilidad",
}
ROLE_NAMES = {
    "market_context": "Contexto de mercado",
    "sentiment": "Sentimiento",
    "strategic_validator": "Validación estratégica",
}
ACTION_NAMES = {"BUY": "Comprar", "HOLD": "Mantener", "SELL": "Vender"}


def percentage(value: float) -> str:
    return f"{value:.2%}".replace(".", ",")


def resolve_data_root() -> tuple[Path, str]:
    override = os.getenv("QQQ_DASHBOARD_DATA_ROOT")
    if override:
        return Path(override), "Demo reproducible"
    local_required = LOCAL_ARTIFACTS / "final_quantitative" / "final_test_metrics.json"
    if local_required.exists():
        return LOCAL_ARTIFACTS, "Resultados locales completos"
    return DEMO_ARTIFACTS, "Demo reproducible"


@st.cache_data(show_spinner=False)
def load_data(root: str, schema_version: str) -> DashboardArtifacts:
    del schema_version  # Its value deliberately invalidates stale Streamlit Cloud cache entries.
    return load_dashboard_artifacts(root)


def equity_figure(equity: pd.DataFrame, hybrid: pd.Series | None = None) -> go.Figure:
    frame = equity.copy()
    if hybrid is not None:
        frame["hybrid_multiagent"] = hybrid
    long = (
        frame.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="strategy", value_name="equity")
    )
    long["strategy"] = long["strategy"].map(EQUITY_NAMES).fillna(long["strategy"])
    figure = px.line(
        long,
        x="date",
        y="equity",
        color="strategy",
        color_discrete_map=COLORS,
        labels={"date": "Fecha", "equity": "Capital normalizado", "strategy": ""},
    )
    figure.update_traces(line_width=2.4)
    figure.update_layout(
        hovermode="x unified",
        legend_orientation="h",
        legend_y=1.12,
        margin=dict(l=10, r=10, t=55, b=10),
        height=440,
    )
    return figure


def comparison_table(metrics: dict[str, object], hybrid: dict[str, float] | None) -> pd.DataFrame:
    comparison: dict[str, dict[str, float]] = {
        "Multiagente cuantitativo": metrics["quantitative_multiagent"]
    }
    if hybrid is not None:
        comparison["Multiagente híbrido"] = hybrid
    for name, values in metrics["baselines"].items():
        comparison[EQUITY_NAMES.get(name, name)] = values
    frame = pd.DataFrame(comparison).T[
        [
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "market_exposure",
        ]
    ]
    return frame.rename(
        columns={
            "cumulative_return": "Rentabilidad acumulada",
            "annualized_return": "Rentabilidad anualizada",
            "annualized_volatility": "Volatilidad",
            "sharpe_ratio": "Sharpe",
            "maximum_drawdown": "Drawdown máximo",
            "market_exposure": "Exposición",
        }
    )


def representative_dates(decisions: pd.DataFrame) -> dict[pd.Timestamp, str]:
    representatives: dict[pd.Timestamp, str] = {}
    action_difference = decisions.loc[
        decisions["action"] != decisions["quantitative_proposed_action"]
    ]
    if not action_difference.empty:
        representatives[action_difference.index[0]] = "Cambio introducido por el comité LLM"
    vetoes = decisions.loc[decisions["risk_veto_triggered"]]
    if not vetoes.empty:
        representatives[vetoes.index[0]] = "Veto de riesgo"
    for action, label in (("BUY", "Compra"), ("SELL", "Venta")):
        matches = decisions.loc[decisions["action"] == action]
        if not matches.empty:
            representatives.setdefault(matches.index[0], label)
    signal_columns = ["technical_signal", "momentum_signal", "risk_signal"]
    disagreement = decisions[signal_columns].std(axis=1).idxmax()
    representatives.setdefault(disagreement, "Máximo desacuerdo cuantitativo")
    return representatives


def render_header(title: str, subtitle: str) -> None:
    st.markdown(f"## {title}")
    st.markdown(f"<p class='page-subtitle'>{subtitle}</p>", unsafe_allow_html=True)


def comparison_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Categoría": "Modelo propio",
                "Técnica": "Multiagente cuantitativo V1",
                "Descripción": "Tres random forests, suma ponderada y veto binario de riesgo.",
            },
            {
                "Categoría": "Modelo propio",
                "Técnica": "Multiagente híbrido V1",
                "Descripción": "V1 más contexto, sentimiento y validación mediante LLM.",
            },
            {
                "Categoría": "Modelo propio",
                "Técnica": "Multiagente calibrado V2",
                "Descripción": "Selección temporal, calibración y exposición 0/50/100 %.",
            },
            {
                "Categoría": "Modelo propio · investigación",
                "Técnica": "Multiagente continuo V3",
                "Descripción": (
                    "Panel cross-asset en shadow mode y riesgo QQQ con efectivo remunerado."
                ),
            },
            {
                "Categoría": "Modelo propio · investigación",
                "Técnica": "Multiagente diario V4",
                "Descripción": (
                    "Riesgo calibrado, volatilidad HAR, tendencia y dirección con abstención."
                ),
            },
            {
                "Categoría": "Benchmark conocido",
                "Técnica": "Buy & Hold",
                "Descripción": "Comprar QQQ al inicio y mantener el 100 % de exposición.",
            },
            {
                "Categoría": "Benchmark conocido",
                "Técnica": "SMA 50/200",
                "Descripción": "Invertir cuando la media de 50 días supera a la de 200.",
            },
            {
                "Categoría": "Benchmark conocido",
                "Técnica": "Regresión logística",
                "Descripción": "Clasificador lineal único sin deliberación multiagente.",
            },
        ]
    )


st.set_page_config(
    page_title="QQQ Multi-Agent Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink: #0f172a; --muted: #64748b; --accent: #0f766e; }
    .stApp { background: linear-gradient(180deg, #f8fafc 0%, #ffffff 35%); }
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: .42rem .65rem; border-radius: .55rem;
    }
    .hero {
        padding: 2.4rem 2.6rem; border-radius: 1.25rem; color: white;
        background: radial-gradient(circle at 82% 18%, #14b8a6 0%, #0f766e 22%, #0f172a 66%);
        box-shadow: 0 18px 55px rgba(15, 23, 42, .16); margin-bottom: 1.4rem;
    }
    .hero h1 { font-size: clamp(2rem, 4vw, 3.35rem); line-height: 1.03; margin: 0 0 .8rem; }
    .hero p { max-width: 760px; color: #dbeafe; font-size: 1.08rem; margin: 0; }
    .eyebrow { letter-spacing: .12em; text-transform: uppercase; font-size: .78rem;
        color: #99f6e4; font-weight: 700; margin-bottom: .7rem; }
    .badge { display: inline-block; padding: .3rem .62rem; margin: .8rem .35rem 0 0;
        border-radius: 999px; background: rgba(255,255,255,.13); font-size: .82rem; }
    .page-subtitle { color: var(--muted); margin-top: -.55rem; margin-bottom: 1.3rem; }
    [data-testid="stMetric"] { background: white; border: 1px solid #e2e8f0;
        padding: .9rem 1rem; border-radius: .8rem; box-shadow: 0 4px 16px rgba(15,23,42,.045); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    div[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255,255,255,.72); }
    .footer { color: #64748b; font-size: .82rem; border-top: 1px solid #e2e8f0;
        margin-top: 2.5rem; padding-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

data_root, data_mode = resolve_data_root()
try:
    artifacts = load_data(str(data_root), DASHBOARD_SCHEMA_VERSION)
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

config = load_config(ROOT / "configs" / "base.yaml")
v2_metrics_bundle = getattr(artifacts, "v2_metrics", None)
v2_decisions_bundle = getattr(artifacts, "v2_decisions", None)
v2_equity_bundle = getattr(artifacts, "v2_equity", None)
v2_model_leaderboard = getattr(artifacts, "v2_model_leaderboard", None)
v2_protected_metrics = getattr(artifacts, "v2_protected_metrics", None)
v2_protected_decisions = getattr(artifacts, "v2_protected_decisions", None)
v2_protected_equity = getattr(artifacts, "v2_protected_equity", None)
v3_metrics_bundle = getattr(artifacts, "v3_metrics", None)
v3_decisions_bundle = getattr(artifacts, "v3_decisions", None)
v3_equity_bundle = getattr(artifacts, "v3_equity", None)
v4_metrics_bundle = getattr(artifacts, "v4_metrics", None)
v4_decisions_bundle = getattr(artifacts, "v4_decisions", None)
v4_equity_bundle = getattr(artifacts, "v4_equity", None)
v8_metrics_bundle = getattr(artifacts, "v8_metrics", None)
v8_decisions_bundle = getattr(artifacts, "v8_decisions", None)
v8_equity_bundle = getattr(artifacts, "v8_equity", None)
v8_current_decision = getattr(artifacts, "v8_current_decision", None)
final_metrics = artifacts.final_quantitative_metrics
final_hybrid_metrics = artifacts.final_hybrid_metrics
final_strategy = (
    final_hybrid_metrics["hybrid_multiagent"]
    if final_hybrid_metrics is not None
    else final_metrics["quantitative_multiagent"]
)
headline_strategy = (
    v2_protected_metrics["v2_multiagent"]
    if v2_protected_metrics is not None
    else final_strategy
)
headline_decisions = (
    len(v8_decisions_bundle)
    if v8_decisions_bundle is not None
    else len(v2_protected_decisions)
    if v2_protected_decisions is not None
    else 105
)
v8_headline = (
    v8_metrics_bundle["subperiods"]["retrospective"]["balanced"]
    if v8_metrics_bundle is not None
    else None
)

st.sidebar.markdown("# ◈ QQQ Multi-Agent Lab")
st.sidebar.caption("Framework jerárquico y explicable")
page = st.sidebar.radio(
    "Navegación",
    (
        "Inicio",
        "Arquitectura",
        "V8 final",
        "Resultados",
        "Diagnóstico V2",
        "Investigación V3",
        "Laboratorio V4",
        "Decisiones",
        "Explicabilidad",
        "Metodología",
    ),
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.markdown(f"**Estado:** {data_mode}")
st.sidebar.caption(
    "V8 · tres perfiles explicables | salida actual en modo paper trading"
)
st.sidebar.success("V8 reproducible y auditada")

if page == "Inicio":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Trabajo Final de Máster · Alex Soler Trias</div>
          <h1>Un comité de agentes que razona, decide y explica</h1>
          <p>Framework experimental sobre QQQ que combina modelos cuantitativos, agentes LLM y
          un coordinador jerárquico. La V8 ofrece perfiles conservador, equilibrado y agresivo,
          conserva cada evidencia y produce una salida utilizable en paper trading.</p>
          <span class="badge">Long-biased</span><span class="badge">Walk-forward purgado</span>
          <span class="badge">25–175 %</span><span class="badge">Tres perfiles de riesgo</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    displayed_headline = v8_headline or headline_strategy
    columns[0].metric(
        "Rentabilidad anual V8 equilibrada",
        percentage(displayed_headline.get("annualized_return", 0.0)),
    )
    columns[1].metric("Sharpe", f"{displayed_headline['sharpe_ratio']:.3f}")
    columns[2].metric(
        "Drawdown máximo", percentage(displayed_headline["maximum_drawdown"])
    )
    columns[3].metric("Decisiones auditadas", str(headline_decisions))

    st.markdown("### Qué demuestra el prototipo")
    demo_columns = st.columns(3)
    with demo_columns[0].container(border=True):
        st.markdown("#### Especialización")
        st.write(
            "Agentes técnicos, de momentum, riesgo y régimen compiten con modelos sencillos y "
            "pueden abstenerse cuando su AUC temporal es insuficiente."
        )
    with demo_columns[1].container(border=True):
        st.markdown("#### Coordinación controlada")
        st.write(
            "Un coordinador determinista transforma probabilidades calibradas en exposición "
            "gradual. El LLM nunca ejecuta operaciones directamente."
        )
    with demo_columns[2].container(border=True):
        st.markdown("#### Trazabilidad")
        st.write(
            "SHAP, LIME, referencias de evidencia y versiones de modelo permiten reconstruir cada "
            "recomendación."
        )

    st.info(
        "Resultado retrospectivo, no garantía prospectiva: el perfil equilibrado mejora la "
        "rentabilidad y reduce el drawdown frente a Buy & Hold en 2023–agosto de 2026. La web "
        "muestra también periodos desfavorables, benchmarks equivalentes e incertidumbre."
    )

elif page == "Arquitectura":
    render_header(
        "Arquitectura jerárquica",
        "Las capas separan predicción, razonamiento contextual, coordinación y control de riesgo.",
    )
    labels = [
        "Tendencia",
        "Volatilidad",
        "Drawdown",
        "Fuerza relativa",
        "Contexto LLM",
        "Validador estratégico",
        "Coordinador",
        "Control de riesgo",
        "Exposición 25–175 %",
    ]
    figure = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                color=["#38bdf8"] * 3 + ["#a78bfa"] * 3 + ["#14b8a6", "#fb7185", "#0f172a"],
                pad=24,
                thickness=22,
            ),
            link=dict(
                source=[0, 1, 2, 3, 4, 5, 6, 7],
                target=[6, 6, 6, 6, 6, 6, 7, 8],
                value=[30, 25, 25, 8, 6, 6, 100, 100],
                color=["rgba(56,189,248,.3)"] * 3
                + ["rgba(167,139,250,.3)"] * 3
                + ["rgba(20,184,166,.35)", "rgba(251,113,133,.35)"],
            ),
        )
    )
    figure.update_layout(height=480, margin=dict(l=20, r=20, t=20, b=20), font_size=13)
    st.plotly_chart(figure, width="stretch")

    layer_columns = st.columns(4)
    descriptions = (
        ("01 · Datos", "QQQ, SPY, VIX y efectivo, fechados sin información futura."),
        ("02 · Agentes", "Tendencia, volatilidad, drawdown y contexto bajo un contrato común."),
        ("03 · Coordinación", "Autoridad acotada y tres perfiles explícitos de riesgo."),
        ("04 · Salida", "Exposición continua, explicación y traza preparada para paper trading."),
    )
    for column, (title, description) in zip(layer_columns, descriptions, strict=True):
        with column.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(description)

elif page == "V8 final":
    render_header(
        "V8 · Comité de inversión explicable",
        "Tres perfiles de exposición a QQQ, benchmarks de riesgo equivalente y salida actual "
        "preparada para paper trading.",
    )
    if (
        v8_metrics_bundle is None
        or v8_equity_bundle is None
        or v8_current_decision is None
    ):
        st.warning("El paquete V8 no está disponible en este conjunto de datos.")
        st.stop()

    profile_labels = {
        "Conservador": "conservative",
        "Equilibrado": "balanced",
        "Agresivo": "aggressive",
    }
    selected_label = st.segmented_control(
        "Perfil de riesgo",
        options=list(profile_labels),
        default="Equilibrado",
    )
    selected_profile = profile_labels[selected_label]
    current = v8_current_decision["profiles"][selected_profile]
    current_columns = st.columns(5)
    current_columns[0].metric(
        "Fecha de la señal",
        pd.Timestamp(v8_current_decision["as_of"]).strftime("%d/%m/%Y"),
    )
    current_columns[1].metric("Exposición propuesta", percentage(current["exposure"]))
    current_columns[2].metric(
        "Acción",
        {"INCREASE": "Aumentar", "REDUCE": "Reducir", "HOLD": "Mantener"}.get(
            current["action"], current["action"]
        ),
    )
    current_columns[3].metric("Régimen", current["state"].replace("_", " ").title())
    current_columns[4].metric(
        "Fuerza de señal", percentage(current["signal_strength"])
    )
    st.info(current["explanation"])
    st.caption(
        "Esta es una salida académica en modo paper trading. No envía órdenes ni constituye "
        "asesoramiento financiero."
    )

    agents = v8_current_decision["agents"]
    st.markdown("### Deliberación del comité")
    agent_columns = st.columns(4)
    with agent_columns[0].container(border=True):
        st.markdown("**Agente de tendencia**")
        st.metric("Puntuación", percentage(agents["trend"]["score"]))
        st.caption("Consenso de medias móviles y momentum a varios horizontes.")
    with agent_columns[1].container(border=True):
        st.markdown("**Agente de volatilidad**")
        st.metric(
            "Volatilidad anual",
            percentage(agents["volatility"]["annualized_forecast"]),
        )
        st.caption("Puede limitar la exposición bajo volatilidad alta o extrema.")
    with agent_columns[2].container(border=True):
        st.markdown("**Agente de drawdown**")
        st.metric("Drawdown 252 sesiones", percentage(agents["drawdown"]["current_252"]))
        st.caption("Señal consultiva: no veta por sí sola para evitar reaccionar tarde.")
    with agent_columns[3].container(border=True):
        st.markdown("**Fuerza relativa**")
        st.metric("QQQ frente a SPY", percentage(agents["relative_strength"]["qqq_vs_spy_20"]))
        st.caption("Contextualiza el régimen, sin autoridad directa sobre la operación.")

    st.markdown("### Frontera histórica de rentabilidad y riesgo")
    selection_metrics = v8_metrics_bundle["subperiods"]["selection"]
    retrospective_metrics = v8_metrics_bundle["subperiods"]["retrospective"]
    benchmark_names = {
        "constant_100": "QQQ Buy & Hold",
        "volatility_target_35": "Volatility target 35 %",
    }
    visible = [selected_profile, "constant_100", "volatility_target_35"]
    comparison_rows = []
    for period_name, period_values in (
        ("Selección 2004–2022", selection_metrics),
        ("Retrospectivo 2023–2026", retrospective_metrics),
    ):
        for name in visible:
            values = period_values[name]
            comparison_rows.append(
                {
                    "Periodo": period_name,
                    "Estrategia": (
                        selected_label if name == selected_profile else benchmark_names[name]
                    ),
                    "Rentabilidad anual": values["annualized_return"],
                    "Volatilidad": values["annualized_volatility"],
                    "Sharpe": values["sharpe_ratio"],
                    "Drawdown máximo": values["maximum_drawdown"],
                    "Exposición media": values["market_exposure"],
                }
            )
    st.dataframe(
        pd.DataFrame(comparison_rows).style.format(
            {
                "Rentabilidad anual": "{:.2%}",
                "Volatilidad": "{:.2%}",
                "Sharpe": "{:.3f}",
                "Drawdown máximo": "{:.2%}",
                "Exposición media": "{:.0%}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    equity = v8_equity_bundle[
        [selected_profile, "buy_and_hold", "volatility_target_35"]
    ].dropna()
    equity = equity.divide(equity.iloc[0])
    equity = equity.rename(
        columns={
            selected_profile: selected_label,
            "buy_and_hold": "QQQ Buy & Hold",
            "volatility_target_35": "Volatility target 35 %",
        }
    )
    equity_long = equity.rename_axis("Fecha").reset_index().melt(
        id_vars="Fecha", var_name="Estrategia", value_name="Capital"
    )
    v8_figure = px.line(
        equity_long,
        x="Fecha",
        y="Capital",
        color="Estrategia",
        log_y=True,
    )
    v8_figure.update_layout(
        height=460,
        hovermode="x unified",
        legend_orientation="h",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(v8_figure, width="stretch")
    st.caption("Escala logarítmica y capital rebased a 1 al comienzo de 2004.")

    st.markdown("### Incertidumbre y control del sobreajuste")
    robustness = v8_metrics_bundle["robustness"]
    bootstrap = robustness["block_bootstrap"][selected_profile]["selection"][
        "vs_buy_and_hold"
    ]
    dsr = robustness["deflated_sharpe"][selected_profile]
    pbo = robustness["candidate_family_cscv"]
    robustness_columns = st.columns(4)
    robustness_columns[0].metric(
        "Diferencia anual media", percentage(bootstrap["mean_difference"])
    )
    robustness_columns[1].metric(
        "P(supera QQQ)", percentage(bootstrap["probability_strategy_outperforms"])
    )
    robustness_columns[2].metric(
        "Sharpe deflactado", percentage(dsr["deflated_sharpe_probability"])
    )
    robustness_columns[3].metric(
        "PBO familia", percentage(pbo["probability_of_backtest_overfitting"])
    )
    st.caption(
        "El intervalo bootstrap del perfil puede incluir cero y el PBO no es bajo. La evidencia "
        "respalda una frontera de riesgo explicable, no una garantía de alpha persistente."
    )

elif page == "Resultados":
    render_header(
        "Resultados experimentales",
        "Comparación fuera de muestra y validación previa con la configuración congelada.",
    )
    with st.expander("Qué modelos y benchmarks estamos comparando", expanded=True):
        st.dataframe(comparison_catalog(), width="stretch", hide_index=True)
        st.caption(
            "Las ablaciones de V2 no son técnicas externas: son variantes internas que eliminan "
            "un componente para medir qué parte del sistema aporta o perjudica el resultado."
        )
    final_tab, validation_tab = st.tabs(["Test final · 2023–2024", "Validación · 2020–2022"])
    with final_tab:
        result_columns = st.columns(5)
        result_columns[0].metric(
            "Rentabilidad acumulada", percentage(final_strategy["cumulative_return"])
        )
        result_columns[1].metric(
            "Rentabilidad anual", percentage(final_strategy["annualized_return"])
        )
        result_columns[2].metric("Volatilidad", percentage(final_strategy["annualized_volatility"]))
        result_columns[3].metric("Sharpe", f"{final_strategy['sharpe_ratio']:.3f}")
        result_columns[4].metric("Exposición", percentage(final_strategy["market_exposure"]))
        st.plotly_chart(
            equity_figure(
                artifacts.final_quantitative_equity,
                artifacts.final_hybrid_strategy["equity"]
                if artifacts.final_hybrid_strategy is not None
                else None,
            ),
            width="stretch",
        )
        table = comparison_table(
            final_metrics,
            final_hybrid_metrics["hybrid_multiagent"] if final_hybrid_metrics is not None else None,
        )
        st.dataframe(
            table.style.format(
                {
                    "Rentabilidad acumulada": "{:.2%}",
                    "Rentabilidad anualizada": "{:.2%}",
                    "Volatilidad": "{:.2%}",
                    "Sharpe": "{:.3f}",
                    "Drawdown máximo": "{:.2%}",
                    "Exposición": "{:.2%}",
                }
            ),
            width="stretch",
        )
        st.warning(
            "El multiagente redujo la volatilidad frente a buy & hold y medias 50/200, pero quedó "
            "por debajo de los tres baselines en rentabilidad, Sharpe y drawdown máximo."
        )
    with validation_tab:
        validation_strategy = artifacts.hybrid_metrics["hybrid_multiagent"]
        columns = st.columns(4)
        columns[0].metric(
            "Rentabilidad acumulada", percentage(validation_strategy["cumulative_return"])
        )
        columns[1].metric(
            "Rentabilidad anual", percentage(validation_strategy["annualized_return"])
        )
        columns[2].metric("Sharpe", f"{validation_strategy['sharpe_ratio']:.3f}")
        columns[3].metric("Drawdown máximo", percentage(validation_strategy["maximum_drawdown"]))
        st.plotly_chart(
            equity_figure(artifacts.equity, artifacts.hybrid_strategy["equity"]),
            width="stretch",
        )
        st.caption(
            "La configuración se seleccionó únicamente con información hasta 2022. El periodo "
            "2023–2024 se abrió después de documentar la congelación experimental."
        )

elif page == "Diagnóstico V2":
    render_header(
        "Diagnóstico y laboratorio V2",
        "Probabilidades calibradas, exposición gradual, ablaciones y control explícito "
        "del test protegido.",
    )
    if (
        v2_metrics_bundle is None
        or v2_decisions_bundle is None
        or v2_equity_bundle is None
    ):
        st.info(
            "La arquitectura V2 está implementada, pero este paquete de demostración todavía no "
            "incluye su ejecución. Puede generarse con `qqq-agents v2-development` sin consultar "
            "el periodo protegido 2025–2026."
        )
    else:
        v2_metrics = v2_metrics_bundle
        v2_strategy = v2_metrics["v2_multiagent"]
        if v2_protected_metrics is not None and v2_protected_equity is not None:
            protected = v2_protected_metrics
            protected_strategy = protected["v2_multiagent"]
            st.success(
                "Test protegido abierto después de congelar configuración y código: "
                "2025–agosto de 2026."
            )
            st.markdown("### Resultado protegido")
            protected_columns = st.columns(5)
            protected_columns[0].metric(
                "Rentabilidad V2", percentage(protected_strategy["cumulative_return"])
            )
            protected_columns[1].metric(
                "Rentabilidad Buy & Hold",
                percentage(protected["baselines"]["buy_and_hold"]["cumulative_return"]),
            )
            protected_columns[2].metric("Sharpe V2", f"{protected_strategy['sharpe_ratio']:.3f}")
            protected_columns[3].metric(
                "Drawdown V2", percentage(protected_strategy["maximum_drawdown"])
            )
            protected_columns[4].metric(
                "Exposición V2", percentage(protected_strategy["market_exposure"])
            )
            protected_curves = v2_protected_equity.loc[
                :,
                [
                    name
                    for name in ("v2_multiagent", "buy_and_hold", "sma_50_200")
                    if name in v2_protected_equity
                ],
            ]
            st.plotly_chart(equity_figure(protected_curves), width="stretch")
            st.info(
                "V2 supera a SMA y reduce el drawdown frente a Buy & Hold, pero no supera la "
                "rentabilidad ni el Sharpe de Buy & Hold. La selección congelada en desarrollo "
                "se conserva y se muestra por separado."
            )
            st.divider()
        else:
            st.warning(
                "2020–2024 es desarrollo, no un test final. El periodo protegido todavía no "
                "se ha incluido en este paquete."
            )
        st.markdown("### Resultado de desarrollo · 2020–2024")
        recommendation = v2_metrics["deployment_recommendation"]
        if recommendation["multiagent_promotion_eligible"]:
            st.success("V2 satisface el criterio predefinido de promoción frente a SMA 50/200.")
        else:
            st.error(
                "Resultado de desarrollo: SMA 50/200 conserva el papel de champion y V2 queda "
                "en shadow mode. La aplicación no oculta ni reajusta este resultado."
            )
        metric_columns = st.columns(5)
        metric_columns[0].metric(
            "Rentabilidad acumulada", percentage(v2_strategy["cumulative_return"])
        )
        metric_columns[1].metric("Sharpe", f"{v2_strategy['sharpe_ratio']:.3f}")
        metric_columns[2].metric(
            "Drawdown máximo", percentage(v2_strategy["maximum_drawdown"])
        )
        metric_columns[3].metric("Exposición", percentage(v2_strategy["market_exposure"]))
        metric_columns[4].metric("Operaciones reales", str(v2_metrics["operation_count"]))

        curves = v2_equity_bundle.loc[
            :,
            [
                name
                for name in ("v2_multiagent", "buy_and_hold", "sma_50_200")
                if name in v2_equity_bundle
            ],
        ]
        st.plotly_chart(equity_figure(curves), width="stretch")

        st.markdown("### V2 frente a benchmarks conocidos")
        benchmark_comparison = {"Multiagente V2 · propio": v2_strategy}
        benchmark_comparison.update(
            {
                EQUITY_NAMES.get(name, name): values
                for name, values in v2_metrics["baselines"].items()
            }
        )
        benchmark_table = pd.DataFrame(benchmark_comparison).T[
            ["cumulative_return", "sharpe_ratio", "maximum_drawdown", "market_exposure"]
        ]
        benchmark_table.columns = ["Rentabilidad", "Sharpe", "Drawdown", "Exposición"]
        table_format = {
            "Rentabilidad": "{:.2%}",
            "Sharpe": "{:.3f}",
            "Drawdown": "{:.2%}",
            "Exposición": "{:.2%}",
        }
        st.dataframe(benchmark_table.style.format(table_format), width="stretch")

        st.markdown("### Ablaciones internas de V2")
        st.caption(
            "No son competidores externos: cada fila desactiva o modifica una parte de nuestro "
            "sistema para identificar qué componente explica el resultado."
        )
        ablations = {
            f"{name.replace('_', ' ')}": values
            for name, values in v2_metrics["ablations"].items()
        }
        ablation_table = pd.DataFrame(ablations).T[
            ["cumulative_return", "sharpe_ratio", "maximum_drawdown", "market_exposure"]
        ]
        ablation_table.columns = ["Rentabilidad", "Sharpe", "Drawdown", "Exposición"]
        st.dataframe(ablation_table.style.format(table_format), width="stretch")

        with st.expander("Cómo interpretar las ablaciones"):
            st.markdown(
                "- **structural long:** mantener QQQ sin señales tácticas.\n"
                "- **risk only:** usar únicamente el agente de riesgo calibrado.\n"
                "- **directional only:** usar únicamente los agentes direccionales.\n"
                "- **binary calibrated risk:** riesgo calibrado, pero posición 0/100 %.\n"
                "- **uncalibrated risk:** misma política usando probabilidades sin calibrar."
            )

        risk_report = v2_metrics["probability_report"]["risk"]
        direction_report = v2_metrics["probability_report"]["direction"]
        diagnostic_columns = st.columns(2)
        with diagnostic_columns[0]:
            st.markdown("### Calibración del riesgo")
            calibration = pd.DataFrame(risk_report["calibration"])
            calibration_figure = go.Figure()
            calibration_figure.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta",
                    line=dict(color="#94a3b8", dash="dash"),
                )
            )
            calibration_figure.add_trace(
                go.Scatter(
                    x=calibration["predicted_probability"],
                    y=calibration["observed_frequency"],
                    mode="lines+markers",
                    name="Agente de riesgo",
                    marker=dict(size=10, color="#fb7185"),
                )
            )
            calibration_figure.update_layout(
                xaxis_title="Probabilidad estimada",
                yaxis_title="Frecuencia observada",
                height=350,
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(calibration_figure, width="stretch")
        with diagnostic_columns[1]:
            st.markdown("### Capacidad predictiva")
            st.metric("AUC direccional", f"{direction_report['auc']:.3f}")
            st.metric("AUC de riesgo", f"{risk_report['auc']:.3f}")
            st.metric("Brier de riesgo", f"{risk_report['brier_score']:.3f}")
            st.metric(
                "Probabilidad de backtest overfitting",
                percentage(
                    v2_metrics["backtest_overfitting"]["probability_of_backtest_overfitting"]
                ),
            )
            st.caption(
                "La calibración puede mejorar la escala de una probabilidad, pero no crea "
                "discriminación. Si el AUC temporal es insuficiente, el agente debe abstenerse."
            )

        yearly_rows = []
        for year, values in v2_metrics["probability_report"]["by_year"].items():
            yearly_rows.append(
                {
                    "Año": year,
                    "AUC dirección": values["direction"]["auc"],
                    "AUC riesgo": values["risk"]["auc"],
                    "Riesgo observado": values["risk"]["positive_rate"],
                    "Riesgo estimado": values["risk"]["mean_probability"],
                }
            )
        st.markdown("### Estabilidad anual")
        st.dataframe(
            pd.DataFrame(yearly_rows).set_index("Año").style.format("{:.3f}"),
            width="stretch",
        )

        if v2_model_leaderboard is not None:
            selected = v2_model_leaderboard.loc[v2_model_leaderboard["selected"]].copy()
            selected["cutoff"] = selected["cutoff"].dt.strftime("%Y-%m-%d")
            st.markdown("### Modelos seleccionados exclusivamente con datos pasados")
            st.dataframe(
                selected[["cutoff", "agent_id", "candidate", "raw_auc", "raw_brier"]],
                width="stretch",
                hide_index=True,
            )

elif page == "Investigación V3":
    render_header(
        "V3 · Investigación prospectiva",
        "Predicción cross-asset en observación, riesgo especializado y exposición continua "
        "con efectivo remunerado.",
    )
    if v3_metrics_bundle is None or v3_equity_bundle is None or v3_decisions_bundle is None:
        st.info(
            "Los componentes V3 están implementados, pero este paquete todavía no contiene "
            "su ejecución reproducible. Genérela con `qqq-agents v3-development`."
        )
    else:
        validation = v3_metrics_bundle["subperiods"]["internal_validation_2025_2026"]
        v3_validation = validation["v3_multiagent"]
        buy_hold_validation = validation["baselines"]["buy_and_hold"]
        st.warning(
            "V3 no es un nuevo resultado final. Sus parámetros se seleccionaron con 2020–2024; "
            "2025–agosto de 2026 actúa como validación interna y el periodo prospectivo comienza "
            "en septiembre de 2026."
        )
        columns = st.columns(5)
        columns[0].metric(
            "Rentabilidad anual V3", percentage(v3_validation["annualized_return"])
        )
        columns[1].metric("Sharpe V3", f"{v3_validation['sharpe_ratio']:.3f}")
        columns[2].metric(
            "Sharpe Buy & Hold", f"{buy_hold_validation['sharpe_ratio']:.3f}"
        )
        columns[3].metric("Drawdown V3", percentage(v3_validation["maximum_drawdown"]))
        columns[4].metric("Exposición", percentage(v3_validation["market_exposure"]))

        validation_curves = v3_equity_bundle.loc[
            "2025-01-01":"2026-08-31",
            [
                name
                for name in ("v3_multiagent", "buy_and_hold", "sma_50_200", "volatility_target")
                if name in v3_equity_bundle
            ],
        ]
        rebased = validation_curves.div(validation_curves.iloc[0])
        st.plotly_chart(equity_figure(rebased), width="stretch")

        st.markdown("### Qué se conserva y qué se descarta")
        findings = pd.DataFrame(
            [
                {
                    "Componente": "Agente direccional cross-asset",
                    "Estado": "Shadow mode",
                    "Evidencia": (
                        "Correlación con el retorno siguiente ≈ 0,04; sin poder de decisión."
                    ),
                },
                {
                    "Componente": "Pooling cross-asset para riesgo",
                    "Estado": "Descartado",
                    "Evidencia": (
                        "Empeoró Sharpe y drawdown incluso incorporando identidad del activo."
                    ),
                },
                {
                    "Componente": "Riesgo especializado QQQ",
                    "Estado": "Activo",
                    "Evidencia": (
                        "AUC temporal aproximada 0,60–0,66 en la mayor parte del desarrollo."
                    ),
                },
                {
                    "Componente": "Efectivo remunerado",
                    "Estado": "Activo",
                    "Evidencia": (
                        "Rentabilidad histórica de letras del Tesoro aplicada al capital libre."
                    ),
                },
            ]
        )
        st.dataframe(findings, hide_index=True, width="stretch")

        st.markdown("### Lectura honesta del resultado")
        st.info(
            "En validación interna V3 mejora ligeramente el Sharpe y el drawdown de Buy & Hold, "
            "pero todavía no alcanza la reducción de drawdown del 20–25 % definida como objetivo. "
            "Por ello continúa como investigación y no sustituye a V2 ni al benchmark."
        )
        probability = v3_decisions_bundle["risk_probability"]
        exposure = v3_decisions_bundle["desired_position"]
        relationship = pd.DataFrame(
            {"Probabilidad de riesgo": probability, "Exposición": exposure}
        ).reset_index()
        risk_figure = px.scatter(
            relationship,
            x="Probabilidad de riesgo",
            y="Exposición",
            color="date",
            color_continuous_scale="Teal",
            labels={"date": "Fecha"},
        )
        risk_figure.update_layout(height=390, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(risk_figure, width="stretch")

elif page == "Laboratorio V4":
    render_header(
        "V4 · Laboratorio diario",
        "Especialistas de riesgo, volatilidad HAR y tendencia; la dirección se abstiene si no "
        "demuestra discriminación temporal.",
    )
    if v4_metrics_bundle is None or v4_equity_bundle is None or v4_decisions_bundle is None:
        st.info(
            "Los componentes V4 están implementados, pero este paquete todavía no contiene "
            "su ejecución reproducible. Genérela con `qqq-agents v4-development`."
        )
    else:
        validation = v4_metrics_bundle["subperiods"]["internal_validation"]
        v4_validation = validation["v4_multiagent"]
        validation_baselines = validation["baselines"]
        buy_hold_validation = validation_baselines["buy_and_hold"]
        st.warning(
            "V4 es investigación, no un nuevo test final. La política se eligió con 2018–2022; "
            "2023–agosto de 2026 es validación interna y septiembre de 2026 permanece cerrado."
        )
        columns = st.columns(5)
        columns[0].metric("Rentabilidad anual", percentage(v4_validation["annualized_return"]))
        columns[1].metric("Sharpe V4", f"{v4_validation['sharpe_ratio']:.3f}")
        columns[2].metric(
            "Sharpe Buy & Hold", f"{buy_hold_validation['sharpe_ratio']:.3f}"
        )
        columns[3].metric("Drawdown máximo", percentage(v4_validation["maximum_drawdown"]))
        columns[4].metric("Exposición media", percentage(v4_validation["market_exposure"]))

        validation_curves = v4_equity_bundle.loc[
            "2023-01-01":"2026-08-31",
            [
                name
                for name in (
                    "v4_multiagent",
                    "buy_and_hold",
                    "sma_50_200",
                    "trailing_volatility_target",
                )
                if name in v4_equity_bundle
            ],
        ]
        rebased = validation_curves.div(validation_curves.iloc[0])
        st.plotly_chart(equity_figure(rebased), width="stretch")

        benchmark_rows = {"Multiagente V4 · propio": v4_validation}
        benchmark_rows.update(
            {
                EQUITY_NAMES.get(name, name): values
                for name, values in validation_baselines.items()
            }
        )
        benchmark_table = pd.DataFrame(benchmark_rows).T[
            [
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "maximum_drawdown",
                "market_exposure",
            ]
        ]
        benchmark_table.columns = [
            "Rentabilidad anual",
            "Volatilidad",
            "Sharpe",
            "Drawdown",
            "Exposición",
        ]
        st.dataframe(
            benchmark_table.style.format(
                {
                    "Rentabilidad anual": "{:.2%}",
                    "Volatilidad": "{:.2%}",
                    "Sharpe": "{:.3f}",
                    "Drawdown": "{:.2%}",
                    "Exposición": "{:.2%}",
                }
            ),
            width="stretch",
        )

        drawdown_improvement = 1 - (
            abs(v4_validation["maximum_drawdown"])
            / abs(buy_hold_validation["maximum_drawdown"])
        )
        retained_return = (
            v4_validation["annualized_return"] / buy_hold_validation["annualized_return"]
        )
        st.info(
            f"Lectura honesta: V4 obtiene el mejor Sharpe de los cuatro métodos y reduce el "
            f"drawdown de Buy & Hold en {percentage(drawdown_improvement)}, pero conserva "
            f"{percentage(retained_return)} de su rentabilidad anual. No supera todas las "
            "técnicas en todas las métricas."
        )

        st.markdown("### Auditoría de los especialistas")
        direction_state = v4_decisions_bundle["direction_active"]
        direction_active_rate = (
            direction_state.mean()
            if direction_state.dtype == bool
            else direction_state.astype(str).str.lower().eq("true").mean()
        )
        agent_findings = pd.DataFrame(
            [
                {
                    "Agente": "Riesgo a cinco sesiones",
                    "Estado": "Activo",
                    "Conclusión": (
                        "AUC temporal anual superior al umbral; ajusta exposición ante colas."
                    ),
                },
                {
                    "Agente": "Dirección a cinco sesiones",
                    "Estado": "Abstención",
                    "Conclusión": (
                        f"Activo en {direction_active_rate:.1%} de las decisiones; no se fuerza "
                        "una señal cuando el AUC es insuficiente."
                    ),
                },
                {
                    "Agente": "Volatilidad HAR",
                    "Estado": "Activo",
                    "Conclusión": (
                        "Estima riesgo futuro; no se interpreta como predicción de dirección."
                    ),
                },
                {
                    "Agente": "Tendencia multi-horizonte",
                    "Estado": "Activo",
                    "Conclusión": "Impone un límite del 70 % solo en tendencia bajista clara.",
                },
            ]
        )
        st.dataframe(agent_findings, hide_index=True, width="stretch")

        chart_columns = st.columns(2)
        daily_view = v4_decisions_bundle.loc["2023-01-01":"2026-08-31"].copy()
        with chart_columns[0]:
            risk_frame = daily_view[
                ["risk_probability", "desired_position"]
            ].rename(
                columns={
                    "risk_probability": "Probabilidad de riesgo",
                    "desired_position": "Exposición",
                }
            )
            risk_figure = px.scatter(
                risk_frame,
                x="Probabilidad de riesgo",
                y="Exposición",
                color="Exposición",
                color_continuous_scale="Teal",
            )
            risk_figure.update_layout(height=360, margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(risk_figure, width="stretch")
        with chart_columns[1]:
            vol_frame = daily_view[
                ["forecast_volatility", "volatility_exposure"]
            ].rename(
                columns={
                    "forecast_volatility": "Volatilidad prevista",
                    "volatility_exposure": "Límite por volatilidad",
                }
            )
            volatility_figure = px.scatter(
                vol_frame,
                x="Volatilidad prevista",
                y="Límite por volatilidad",
                color="Volatilidad prevista",
                color_continuous_scale="Blues",
            )
            volatility_figure.update_layout(
                height=360, margin=dict(l=10, r=10, t=25, b=10)
            )
            st.plotly_chart(volatility_figure, width="stretch")

        robustness = v4_metrics_bundle["robustness"]
        dsr = robustness["deflated_sharpe"]
        pbo = robustness["candidate_family_cscv"]
        st.markdown("### Control del sobreajuste")
        robustness_columns = st.columns(3)
        robustness_columns[0].metric(
            "Configuraciones evaluadas",
            f"{v4_metrics_bundle['configuration_evaluations']:,}".replace(",", "."),
        )
        robustness_columns[1].metric(
            "Probabilidad Sharpe deflactado",
            percentage(dsr["deflated_sharpe_probability"]),
        )
        robustness_columns[2].metric(
            "PBO familia final",
            percentage(pbo["probability_of_backtest_overfitting"]),
        )
        st.caption(
            "El DSR penaliza las 1.660 evaluaciones. El PBO cubre únicamente las ocho series "
            "finales exportadas y no debe interpretarse como una garantía global."
        )
        st.warning(
            "Los diagnósticos estadísticos no confirman una ventaja persistente: el Sharpe "
            f"deflactado solo alcanza una probabilidad del "
            f"{percentage(dsr['deflated_sharpe_probability'])} y el PBO de la familia final es "
            f"{percentage(pbo['probability_of_backtest_overfitting'])}. La conclusión correcta "
            "es candidato prospectivo, no superioridad."
        )
        bootstrap_rows = []
        for name, values in robustness["validation_block_bootstrap"].items():
            bootstrap_rows.append(
                {
                    "Benchmark": EQUITY_NAMES.get(name, name),
                    "Diferencia anual media": values["mean_difference"],
                    "IC 95 % inferior": values["ci_95_lower"],
                    "IC 95 % superior": values["ci_95_upper"],
                    "P(V4 supera)": values["probability_strategy_outperforms"],
                }
            )
        st.dataframe(
            pd.DataFrame(bootstrap_rows).style.format(
                {
                    "Diferencia anual media": "{:.2%}",
                    "IC 95 % inferior": "{:.2%}",
                    "IC 95 % superior": "{:.2%}",
                    "P(V4 supera)": "{:.1%}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Bootstrap circular de bloques de 20 sesiones y 5.000 remuestreos. Compara "
            "rentabilidad anual, no Sharpe ni drawdown; los tres intervalos incluyen cero."
        )

elif page == "Decisiones":
    render_header(
        "Explorador de decisiones",
        "Reconstrucción de señales, contribuciones, evidencias y acción final para cada semana.",
    )
    quantitative = artifacts.final_quantitative_decisions
    hybrid = artifacts.final_hybrid_decisions
    representatives = representative_dates(hybrid)
    mode = st.segmented_control(
        "Selección",
        options=("Casos representativos", "Todas las fechas"),
        default="Casos representativos",
    )
    dates = list(representatives) if mode == "Casos representativos" else list(hybrid.index[::-1])

    def format_date(value: pd.Timestamp) -> str:
        label = value.strftime("%d/%m/%Y")
        return f"{label} · {representatives[value]}" if value in representatives else label

    selected_date = st.selectbox("Fecha de decisión", dates, format_func=format_date)
    hybrid_row = hybrid.loc[selected_date]
    quant_row = quantitative.loc[selected_date]
    prior_positions = hybrid["desired_position"].shift(1).fillna(0.0)
    executed = bool(
        hybrid.loc[selected_date, "desired_position"] != prior_positions.loc[selected_date]
    )
    summary = st.columns(6)
    summary[0].metric(
        "Acción cuantitativa", ACTION_NAMES[hybrid_row["quantitative_proposed_action"]]
    )
    summary[1].metric("Acción híbrida", ACTION_NAMES[hybrid_row["action"]])
    summary[2].metric("Puntuación", f"{hybrid_row['score_before_veto']:.3f}")
    summary[3].metric("Posición", "Invertido" if hybrid_row["desired_position"] else "Efectivo")
    summary[4].metric("Personalidad", str(hybrid_row["personality"]).capitalize())
    summary[5].metric("Operación efectiva", "Sí" if executed else "No")

    if hybrid_row["action"] != hybrid_row["quantitative_proposed_action"]:
        st.info("El comité LLM modificó la recomendación cuantitativa en esta fecha.")
    if bool(hybrid_row["risk_veto_triggered"]):
        st.warning("El agente de riesgo activó el veto y bloqueó una nueva exposición.")

    st.caption(
        "El dictamen BUY/HOLD/SELL se muestra por separado de la operación efectiva. Una señal "
        "BUY repetida mientras la cartera ya está invertida no genera una nueva transacción."
    )

    contribution_columns = [
        "technical_contribution",
        "momentum_contribution",
        "risk_contribution",
        "market_context_contribution",
        "sentiment_contribution",
        "strategic_validator_contribution",
    ]
    contributions = pd.DataFrame(
        {
            "Agente": [
                ROLE_NAMES.get(
                    name.removesuffix("_contribution"),
                    name.removesuffix("_contribution").replace("_", " ").title(),
                )
                for name in contribution_columns
            ],
            "Contribución": [hybrid_row[name] for name in contribution_columns],
        }
    )
    contribution_figure = px.bar(
        contributions,
        x="Agente",
        y="Contribución",
        color="Contribución",
        color_continuous_scale="RdYlGn",
        range_color=[-0.15, 0.15],
    )
    contribution_figure.update_coloraxes(showscale=False)
    contribution_figure.update_layout(height=350, margin=dict(l=10, r=10, t=25, b=10))
    st.plotly_chart(contribution_figure, width="stretch")

    st.markdown("### Deliberación de los agentes LLM")
    for role in ROLE_NAMES:
        with st.expander(ROLE_NAMES[role], expanded=role == "strategic_validator"):
            st.write(hybrid_row[f"{role}_justification"])
            detail_columns = st.columns(3)
            detail_columns[0].metric("Señal", f"{hybrid_row[f'{role}_signal']:.3f}")
            detail_columns[1].metric("Confianza", f"{hybrid_row[f'{role}_confidence']:.3f}")
            detail_columns[2].metric("Modelo", hybrid_row[f"{role}_model"])
            evidence = json.loads(hybrid_row[f"{role}_evidence_ids"])
            limitations = json.loads(hybrid_row[f"{role}_limitations"])
            st.caption(
                "Evidencias: " + (", ".join(evidence) if evidence else "sin evidencia textual")
            )
            if limitations:
                st.caption("Limitaciones: " + " · ".join(limitations))

    st.markdown("### Explicación SHAP de los agentes cuantitativos")
    tabs = st.tabs(["Técnico", "Momentum", "Riesgo"])
    for tab, agent in zip(tabs, ("technical", "momentum", "risk"), strict=True):
        with tab:
            values = json.loads(quant_row[f"{agent}_shap"])
            shap_frame = pd.DataFrame(values).sort_values("contribution")
            shap_figure = px.bar(
                shap_frame,
                x="contribution",
                y="feature",
                color="contribution",
                color_continuous_scale="RdYlGn",
                orientation="h",
                labels={"contribution": "Contribución SHAP", "feature": "Variable"},
            )
            shap_figure.update_coloraxes(showscale=False)
            st.plotly_chart(shap_figure, width="stretch")

elif page == "Explicabilidad":
    render_header(
        "Explicabilidad por diseño",
        "Cada naturaleza de agente utiliza un mecanismo de explicación apropiado a su salida.",
    )
    explanation_columns = st.columns(3)
    with explanation_columns[0].container(border=True):
        st.markdown("#### SHAP")
        st.write("Atribuciones locales para los random forests técnico, de momentum y de riesgo.")
    with explanation_columns[1].container(border=True):
        st.markdown("#### LIME")
        st.write("Contraste local para casos representativos de compra, venta, veto y desacuerdo.")
    with explanation_columns[2].container(border=True):
        st.markdown("#### Evidencia LLM")
        st.write("Justificación breve, identificadores permitidos y limitaciones declaradas.")

    st.markdown("### Casos representativos explicados con LIME")
    cases = pd.DataFrame(artifacts.lime_cases["cases"])
    selection = artifacts.lime_cases["selection"]
    labels = {
        "strongest_buy": "Compra más intensa",
        "strongest_sell": "Venta más intensa",
        "highest_risk_veto": "Mayor veto de riesgo",
        "largest_disagreement": "Mayor desacuerdo",
    }
    selected_case = st.selectbox("Caso", list(selection), format_func=lambda value: labels[value])
    matching = cases.loc[cases["case_type"] == selected_case]
    first = matching.iloc[0]
    case_columns = st.columns(3)
    case_columns[0].metric("Fecha", first["date"])
    case_columns[1].metric("Acción", ACTION_NAMES[first["action"]])
    case_columns[2].metric("Puntuación", f"{first['score_before_veto']:.3f}")
    for _, case in matching.iterrows():
        with st.expander(f"Agente {case['agent_id']}", expanded=case["agent_id"] == "technical"):
            frame = pd.DataFrame(case["attributions"]).sort_values("contribution")
            figure = px.bar(
                frame,
                x="contribution",
                y="feature",
                color="contribution",
                color_continuous_scale="RdYlGn",
                orientation="h",
                labels={"contribution": "Contribución LIME", "feature": "Variable"},
            )
            figure.update_coloraxes(showscale=False)
            st.plotly_chart(figure, width="stretch")

elif page == "Metodología":
    render_header(
        "Metodología y reproducibilidad",
        "Separación temporal estricta, configuración versionada y límites explícitos.",
    )
    timeline = st.columns(3)
    with timeline[0].container(border=True):
        st.markdown("**1999–2003 · Warm-up**")
        st.caption("Construcción de indicadores; no participa en la comparación.")
    with timeline[1].container(border=True):
        st.markdown("**2004–2022 · Selección**")
        st.caption("Definición de perfiles y evaluación por bloques cronológicos.")
    with timeline[2].container(border=True):
        st.markdown("**2023–ago. 2026 · Retrospectivo**")
        st.caption("Periodo ya observado; no se presenta como holdout prospectivo.")

    st.markdown("### Configuración del coordinador")
    weights = pd.DataFrame(
        {
            "Agente": list(config.coordinator.weights),
            "Peso": list(config.coordinator.weights.values()),
        }
    )
    weights["Agente"] = (
        weights["Agente"]
        .map(ROLE_NAMES)
        .fillna(weights["Agente"].str.replace("_", " ").str.title())
    )
    weight_figure = px.bar(
        weights,
        x="Peso",
        y="Agente",
        orientation="h",
        color="Peso",
        color_continuous_scale="Tealgrn",
    )
    weight_figure.update_coloraxes(showscale=False)
    weight_figure.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(weight_figure, width="stretch")

    method_columns = st.columns(2)
    with method_columns[0]:
        st.markdown("### Garantías aplicadas")
        st.markdown(
            "- Auditoría de fechas máximas de entrenamiento.\n"
            "- Coste de 10 puntos básicos por cambio de exposición.\n"
            "- Caché determinista de respuestas LLM.\n"
            "- Validación de referencias contra el paquete de entrada.\n"
            "- El LLM no ejecuta operaciones directamente."
        )
    with method_columns[1]:
        st.markdown("### Limitaciones")
        st.markdown(
            "- Sin noticias históricas verificadas en esta versión.\n"
            "- Decisión centrada en QQQ; no selecciona acciones individuales.\n"
            "- Exposición sintética: no replica exactamente un ETF apalancado.\n"
            "- Los intervalos de incertidumbre pueden incluir cero.\n"
            "- Uso académico y paper trading; no ejecuta capital real."
        )

    st.markdown("### Coste y reproducción")
    cost_columns = st.columns(4)
    cost_columns[0].metric("Piloto", "$0,008")
    cost_columns[1].metric("Validación", "$0,602")
    cost_columns[2].metric("Test final", "$0,401")
    cost_columns[3].metric("Total estimado", "$1,011")
    st.code(
        "uv sync --extra dashboard\nuv run streamlit run app/streamlit_app.py",
        language="bash",
    )

st.markdown(
    "<div class='footer'>Artefacto académico experimental · No constituye asesoramiento "
    "financiero ni está preparado para operar con capital real.</div>",
    unsafe_allow_html=True,
)
