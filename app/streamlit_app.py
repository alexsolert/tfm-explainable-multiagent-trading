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

COLORS = {
    "Multiagente V2": "#0f766e",
    "Multiagente híbrido": "#14b8a6",
    "Multiagente cuantitativo": "#38bdf8",
    "Buy & hold": "#f59e0b",
    "Medias 50/200": "#a78bfa",
    "Agente logístico único": "#fb7185",
}
EQUITY_NAMES = {
    "v2_multiagent": "Multiagente V2",
    "hybrid_multiagent": "Multiagente híbrido",
    "multiagent": "Multiagente cuantitativo",
    "buy_and_hold": "Buy & hold",
    "sma_50_200": "Medias 50/200",
    "single_logistic_agent": "Agente logístico único",
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
def load_data(root: str) -> DashboardArtifacts:
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
    artifacts = load_data(str(data_root))
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

config = load_config(ROOT / "configs" / "base.yaml")
final_metrics = artifacts.final_quantitative_metrics
final_hybrid_metrics = artifacts.final_hybrid_metrics
final_strategy = (
    final_hybrid_metrics["hybrid_multiagent"]
    if final_hybrid_metrics is not None
    else final_metrics["quantitative_multiagent"]
)
headline_strategy = (
    artifacts.v2_protected_metrics["v2_multiagent"]
    if artifacts.v2_protected_metrics is not None
    else final_strategy
)
headline_decisions = (
    len(artifacts.v2_protected_decisions)
    if artifacts.v2_protected_decisions is not None
    else 105
)

st.sidebar.markdown("# ◈ QQQ Multi-Agent Lab")
st.sidebar.caption("Framework jerárquico y explicable")
page = st.sidebar.radio(
    "Navegación",
    (
        "Inicio",
        "Arquitectura",
        "Resultados",
        "Diagnóstico V2",
        "Decisiones",
        "Explicabilidad",
        "Metodología",
    ),
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.markdown(f"**Estado:** {data_mode}")
st.sidebar.caption("V1 · 2023–2024 | V2 protegida · 2025–2026")
st.sidebar.success("V2 evaluada sin reajuste posterior")

if page == "Inicio":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Trabajo Final de Máster · Alex Soler Trias</div>
          <h1>Un comité de agentes que razona, decide y explica</h1>
          <p>Framework experimental sobre QQQ que combina modelos cuantitativos, agentes LLM y
          un coordinador jerárquico. La V2 calibra el riesgo, admite exposición gradual y conserva
          cada evidencia para reconstruir las decisiones de extremo a extremo.</p>
          <span class="badge">Long-biased</span><span class="badge">Walk-forward purgado</span>
          <span class="badge">0 / 50 / 100 %</span><span class="badge">Test protegido</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    columns[0].metric(
        "Rentabilidad V2 protegida", percentage(headline_strategy["cumulative_return"])
    )
    columns[1].metric("Sharpe", f"{headline_strategy['sharpe_ratio']:.3f}")
    columns[2].metric("Drawdown máximo", percentage(headline_strategy["maximum_drawdown"]))
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
        "Hallazgo protegido: V2 redujo el drawdown frente a Buy & Hold y superó a SMA 50/200, "
        "pero no superó la rentabilidad ni el Sharpe de Buy & Hold. El resultado se muestra sin "
        "reajustes posteriores."
    )

elif page == "Arquitectura":
    render_header(
        "Arquitectura jerárquica",
        "Las capas separan predicción, razonamiento contextual, coordinación y control de riesgo.",
    )
    labels = [
        "Técnico",
        "Momentum",
        "Riesgo",
        "Contexto",
        "Sentimiento",
        "Validador",
        "Coordinador",
        "Veto de riesgo",
        "BUY / HOLD / SELL",
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
        ("01 · Datos", "Variables técnicas de QQQ fechadas y preparadas sin información futura."),
        ("02 · Agentes", "Señal, confianza, explicación y evidencia bajo un contrato común."),
        ("03 · Coordinación", "Suma ponderada, umbrales de acción y personalidad conservadora."),
        ("04 · Salida", "Posición binaria long-only y traza completa de la deliberación."),
    )
    for column, (title, description) in zip(layer_columns, descriptions, strict=True):
        with column.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(description)

elif page == "Resultados":
    render_header(
        "Resultados experimentales",
        "Comparación fuera de muestra y validación previa con la configuración congelada.",
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
        artifacts.v2_metrics is None
        or artifacts.v2_decisions is None
        or artifacts.v2_equity is None
    ):
        st.info(
            "La arquitectura V2 está implementada, pero este paquete de demostración todavía no "
            "incluye su ejecución. Puede generarse con `qqq-agents v2-development` sin consultar "
            "el periodo protegido 2025–2026."
        )
    else:
        v2_metrics = artifacts.v2_metrics
        v2_strategy = v2_metrics["v2_multiagent"]
        if artifacts.v2_protected_metrics is not None and artifacts.v2_protected_equity is not None:
            protected = artifacts.v2_protected_metrics
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
            protected_curves = artifacts.v2_protected_equity.loc[
                :,
                [
                    name
                    for name in ("v2_multiagent", "buy_and_hold", "sma_50_200")
                    if name in artifacts.v2_protected_equity
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

        curves = artifacts.v2_equity.loc[
            :,
            [
                name
                for name in ("v2_multiagent", "buy_and_hold", "sma_50_200")
                if name in artifacts.v2_equity
            ],
        ]
        st.plotly_chart(equity_figure(curves), width="stretch")

        st.markdown("### Qué componente aporta valor")
        comparison = {"Multiagente V2": v2_strategy}
        comparison.update(
            {
                EQUITY_NAMES.get(name, name): values
                for name, values in v2_metrics["baselines"].items()
            }
        )
        comparison.update(
            {
                f"Ablación · {name.replace('_', ' ')}": values
                for name, values in v2_metrics["ablations"].items()
            }
        )
        ablation_table = pd.DataFrame(comparison).T[
            ["cumulative_return", "sharpe_ratio", "maximum_drawdown", "market_exposure"]
        ]
        ablation_table.columns = ["Rentabilidad", "Sharpe", "Drawdown", "Exposición"]
        st.dataframe(
            ablation_table.style.format(
                {
                    "Rentabilidad": "{:.2%}",
                    "Sharpe": "{:.3f}",
                    "Drawdown": "{:.2%}",
                    "Exposición": "{:.2%}",
                }
            ),
            width="stretch",
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

        if artifacts.v2_model_leaderboard is not None:
            selected = artifacts.v2_model_leaderboard.loc[
                artifacts.v2_model_leaderboard["selected"]
            ].copy()
            selected["cutoff"] = selected["cutoff"].dt.strftime("%Y-%m-%d")
            st.markdown("### Modelos seleccionados exclusivamente con datos pasados")
            st.dataframe(
                selected[["cutoff", "agent_id", "candidate", "raw_auc", "raw_brier"]],
                width="stretch",
                hide_index=True,
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
        st.markdown("**2015–2019 · Entrenamiento inicial**")
        st.caption("Solo información anterior a cada predicción.")
    with timeline[1].container(border=True):
        st.markdown("**2020–2022 · Validación**")
        st.caption("Walk-forward expansivo y congelación posterior.")
    with timeline[2].container(border=True):
        st.markdown("**2023–2024 · Test final**")
        st.caption("Apertura única y resultados sin reajuste.")

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
            "- Un único activo y régimen de mercado limitado.\n"
            "- Sin posiciones cortas ni apalancamiento.\n"
            "- No se demuestra superioridad frente a los baselines.\n"
            "- Uso académico; no es un sistema de inversión."
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
