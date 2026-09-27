# Framework multiagente explicable para QQQ

Implementacion experimental del TFM **Framework multi-agente jerarquico y explicable
para la toma de decisiones de trading sobre un ETF del Nasdaq**.

El proyecto estudia una arquitectura long-only sobre QQQ en la que agentes cuantitativos
y agentes basados en modelos de lenguaje producen contribuciones especializadas. Un
coordinador determinista agrega sus señales, aplica un veto de riesgo y conserva una traza
reproducible de cada decision.

## Estado

El repositorio se encuentra en la primera fase del MVP. La especificacion experimental esta
congelada en [`docs/experiment_spec.md`](docs/experiment_spec.md) y se revisara antes de
consultar el periodo final de prueba.

## Principios de diseno

- Separacion entre datos, agentes, coordinacion, backtesting, explicabilidad e interfaz.
- Contrato comun para poder sustituir modelos sin modificar el resto del sistema.
- Configuracion fuera del codigo.
- Disciplina temporal y pruebas explicitas contra look-ahead bias.
- Coordinador reproducible: el LLM no ejecuta directamente operaciones.
- Resultados de LLM versionados y almacenados en cache.

## Instalacion de desarrollo

Se recomienda Python 3.12 y `uv`:

```bash
uv sync --extra dev
uv run pytest
```

Los componentes pesados se instalan solo cuando se necesitan:

```bash
uv sync --extra dev --extra xai
uv sync --extra dev --extra llm --extra dashboard
```

## Primeros comandos

```bash
# Descargar el historico diario de QQQ
uv run qqq-agents download

# Generar el conjunto de caracteristicas y objetivos
uv run qqq-agents prepare

# Comprobar los baselines solo durante el periodo de desarrollo (hasta 2022)
uv run qqq-agents baselines

# Entrenar los tres agentes cuantitativos sin consultar 2023-2024
uv run qqq-agents train-quant

# Validacion walk-forward expansiva sobre 2020-2022
uv run qqq-agents walk-forward

# Repetir la validacion conservando explicaciones SHAP locales
uv run qqq-agents walk-forward --with-shap

# Comparar una personalidad alternativa sin cambiar codigo
uv run qqq-agents walk-forward --personality aggressive

# Generar LIME para compra, venta, veto y desacuerdo representativos
uv run qqq-agents lime-cases

# Abrir el dashboard local
uv run streamlit run app/streamlit_app.py
```

Los datos generados no se versionan. Los comandos y la configuracion permiten reconstruirlos.

## Estructura

```text
configs/                 Parametros experimentales y personalidades
data/                    Datos locales no versionados
docs/                    Especificacion y decisiones metodologicas
src/qqq_agents/          Codigo fuente instalable
tests/                   Pruebas automatizadas
artifacts/               Modelos, trazas, metricas y figuras generadas
```

## Advertencia

Este repositorio tiene finalidad academica y no constituye asesoramiento financiero ni un
sistema preparado para operar con capital real.
