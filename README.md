# Framework multiagente explicable para QQQ

Implementacion experimental del TFM **Framework multi-agente jerarquico y explicable
para la toma de decisiones de trading sobre un ETF del Nasdaq**.

El proyecto estudia una arquitectura long-only sobre QQQ en la que agentes cuantitativos
y agentes basados en modelos de lenguaje producen contribuciones especializadas. Un
coordinador determinista agrega sus señales, aplica un veto de riesgo y conserva una traza
reproducible de cada decision.

## Estado

El MVP esta implementado. La especificacion experimental se congelo antes de abrir una sola vez
el test final 2023-2024. El estado de cumplimiento se resume en
[`docs/mvp_status.md`](docs/mvp_status.md) y los resultados finales se documentan en
[`docs/experiments/2026-09-27-final-test.md`](docs/experiments/2026-09-27-final-test.md).

La V2 se desarrolla en paralelo sin reescribir V1. Añade calibración temporal, selección de modelos,
datos de régimen, posiciones 0/50/100 %, ablaciones y un test 2025–2026 protegido por hash. Su
protocolo se encuentra en [`docs/experiments/v2-protocol.md`](docs/experiments/v2-protocol.md).
El resultado de apertura única se documenta en
[`docs/experiments/2026-09-28-v2-protected-test.md`](docs/experiments/2026-09-28-v2-protected-test.md).

La V3 es una línea prospectiva independiente: añade entrenamiento cross-asset para el agente
direccional, mantiene el riesgo especializado en QQQ, remunera el efectivo, permite exposición
continua y compara también contra un benchmark con objetivo de volatilidad. El protocolo y los
intentos descartados se documentan en
[`docs/experiments/v3-protocol.md`](docs/experiments/v3-protocol.md).

La V4 estudia decisiones diarias con especialistas separados de riesgo, volatilidad HAR,
tendencia y dirección. Compara modelos lineales y no lineales con selección temporal, permite
abstención, registra 1.660 evaluaciones de configuración y reserva septiembre de 2026 en adelante
como periodo prospectivo. La revisión académica y el protocolo están en
[`docs/research/v4-literature-review.md`](docs/research/v4-literature-review.md) y
[`docs/experiments/v4-daily-protocol.md`](docs/experiments/v4-daily-protocol.md). Los resultados y
sus advertencias estadísticas se registran en
[`docs/experiments/2026-09-28-v4-internal-validation.md`](docs/experiments/2026-09-28-v4-internal-validation.md).

La V5 mantiene V4 congelada y separa frecuencias: riesgo y volatilidad diarios, tendencia
semanal y cinco políticas de coordinación comparadas sobre una frontera de Pareto. Sustituye el
evento fijo por targets normalizados a la volatilidad en horizontes de 5, 10 y 20 sesiones. El
protocolo y la evaluación retrospectiva están documentados en
[`docs/experiments/v5-multifrequency-protocol.md`](docs/experiments/v5-multifrequency-protocol.md)
y
[`docs/experiments/2026-09-28-v5-retrospective-assessment.md`](docs/experiments/2026-09-28-v5-retrospective-assessment.md).
Las métricas, auditorías de modelos y frontera de políticas se versionan de forma compacta en
[`research_results/v5`](research_results/v5), sin incluir el histórico de mercado.

La V6 amplía el histórico hasta 1999, incorpora un agente de retorno multi-horizonte y estudia
exposición gestionada entre el 25 % y el 125 %. Compara diez políticas con financiación y costes
explícitos, y mantiene una ablación de tendencia para comprobar si los modelos de retorno añaden
valor incremental. El diseño se describe en
[`docs/experiments/v6-risk-managed-exposure-protocol.md`](docs/experiments/v6-risk-managed-exposure-protocol.md)
y sus resultados en
[`docs/experiments/2026-09-28-v6-retrospective-assessment.md`](docs/experiments/2026-09-28-v6-retrospective-assessment.md).

La V7 explora una frontera de crecimiento más ambiciosa, motivada por la literatura sobre
tendencia, volatility scaling y apalancamiento condicionado al régimen. Predeclara dos perfiles
con exposición máxima del 150 % y 175 %, los compara tanto con Buy & Hold como con controles de
apalancamiento constante y conserva 2004–2022 como periodo exclusivo de selección. La revisión de
evidencia, las limitaciones de comparabilidad y los resultados se documentan en
[`docs/research/v7-internet-evidence.md`](docs/research/v7-internet-evidence.md) y
[`research_results/v7`](research_results/v7).

La V8 consolida el proyecto en un framework final con perfiles conservador, equilibrado y
agresivo, autoridad explícita por agente, controles de exposición equivalente y una salida
auditable preparada para paper trading. El protocolo se encuentra en
[`docs/experiments/v8-final-framework-protocol.md`](docs/experiments/v8-final-framework-protocol.md)
y el paquete compacto de resultados en [`research_results/v8`](research_results/v8).

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

# Validar gratis la ruta completa de los tres agentes LLM
uv run qqq-agents llm-dry-run

# Tras crear .env con OPENAI_API_KEY, ejecutar un unico piloto real acotado
uv run qqq-agents llm-pilot

# Validar primero las 157 fechas con el proveedor local gratuito
uv run qqq-agents hybrid-validation

# Ejecutar la validacion LLM real solo tras aprobar payload y coste
uv run qqq-agents hybrid-validation --provider openai

# Reproducir el test final exige confirmar que la especificacion esta congelada
uv run qqq-agents final-quantitative --confirm-frozen-spec --with-shap
uv run qqq-agents hybrid-final-test --confirm-frozen-spec --provider openai

# Abrir el dashboard local
uv run streamlit run app/streamlit_app.py

# Descargar contexto externo solo hasta 2024 y preparar V2
uv run qqq-agents v2-download-context
uv run qqq-agents v2-prepare

# Ejecutar el periodo de desarrollo 2020–2024 sin abrir el test protegido
uv run qqq-agents v2-development

# Congelar V2 únicamente cuando código, tests y protocolo estén cerrados
uv run qqq-agents v2-freeze

# Solo después de congelar: descargar y evaluar el snapshot protegido
uv run qqq-agents v2-open-protected-data --confirm-frozen-spec
uv run qqq-agents v2-protected-test --confirm-frozen-spec

# Construir y evaluar V3 sin abrir su periodo prospectivo
uv run qqq-agents v3-download
uv run qqq-agents v3-prepare
uv run qqq-agents v3-development

# Construir y evaluar la investigación diaria V4
uv run qqq-agents v4-download
uv run qqq-agents v4-prepare
uv run qqq-agents v4-development

# Construir, evaluar y congelar la investigación multi-frecuencia V5
uv run qqq-agents v5-prepare
uv run qqq-agents v5-development
uv run qqq-agents v5-freeze

# Generar una señal prospectiva sin ejecutar operaciones
uv run qqq-agents v5-shadow --through 2026-09-29
```

El comando `v2-protected-test` exige confirmación explícita y comprueba que el hash de
`configs/v2.yaml` coincide con el registrado durante la congelación. Los datos 2023–2024 ya se han
utilizado para diagnosticar V1 y, por tanto, forman parte del desarrollo de V2, no de su test final.

## Demo web

La aplicacion incluye un conjunto ligero de resultados congelados en `demo_data/`. Por ello, puede
abrirse nada mas clonar el repositorio sin descargar QQQ, reentrenar modelos ni configurar una
clave de API. La interfaz ofrece resultados V1/V2, laboratorios V3/V4, arquitectura, explorador
de decisiones, explicabilidad y metodología.

```bash
uv sync --extra dashboard
uv run streamlit run app/streamlit_app.py
```

Si existen artefactos locales completos, la aplicacion los utiliza de forma preferente. Para
forzar el paquete publico de demostracion:

```bash
QQQ_DASHBOARD_DATA_ROOT=demo_data uv run streamlit run app/streamlit_app.py
```

`requirements.txt` y `.streamlit/config.toml` dejan preparada la aplicacion para un despliegue de
Streamlit basado en GitHub. La demo publica no realiza llamadas a OpenAI ni contiene credenciales.

Los datos generados no se versionan. Los comandos y la configuracion permiten reconstruirlos.
El piloto LLM solo utiliza decisiones del periodo de validacion, almacena las respuestas en cache
y aplica el limite de gasto configurado antes de realizar llamadas. La accion final sigue siendo
responsabilidad del coordinador determinista.
Cuando no existen noticias historicas con marcas temporales verificables, una compuerta de
evidencia fija el sentimiento como neutral sin efectuar una llamada de pago. La integracion de
noticias queda documentada como mejora posterior.

## Estructura

```text
configs/                 Parametros experimentales y personalidades
data/                    Datos locales no versionados
docs/                    Especificacion y decisiones metodologicas
src/qqq_agents/          Codigo fuente instalable
tests/                   Pruebas automatizadas
artifacts/               Modelos, trazas, metricas y figuras generadas
demo_data/               Resultados congelados y seguros para la demo publica
```

## Advertencia

Este repositorio tiene finalidad academica y no constituye asesoramiento financiero ni un
sistema preparado para operar con capital real.
