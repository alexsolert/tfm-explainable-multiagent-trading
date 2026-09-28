# Resultados V5 multi-frecuencia

## Estado experimental

La política se seleccionó exclusivamente con 2018–2022. El intervalo 2023–31 de agosto de 2026 es una evaluación retrospectiva y no un holdout, dado que ya había sido analizado durante V4. V5 se congela antes de iniciar su registro prospectivo el 29 de septiembre de 2026. Ninguna señal de shadow mode autoriza ejecución real.

## Selección 2018–2022

La frontera de Pareto incluyó `risk_only`, `trend_only`, `minimum_caps` y `confirmation`. La máquina de estados asimétrica quedó dominada. Ninguna política cumplió simultáneamente los objetivos de conservar el 95 % de la rentabilidad de Buy & Hold, reducir su drawdown un 20 % y no empeorar su Sharpe. Este incumplimiento se conserva en los artefactos.

`trend_only` fue seleccionada mediante la regla predefinida. Obtuvo rentabilidad anualizada del 12,14 %, Sharpe 0,617 y drawdown −31,01 %, frente a 11,69 %, 0,554 y −35,12 % de Buy & Hold. Conservó más del 100 % de la rentabilidad y redujo la magnitud del drawdown un 11,70 %, por debajo del objetivo del 20 %. `minimum_caps` produjo un Sharpe ligeramente superior, 0,619, pero menor rentabilidad, mayor rotación y el mismo drawdown.

## Evaluación retrospectiva 2023–agosto de 2026

La política congelable `trend_only` obtuvo rentabilidad anualizada del 30,39 %, volatilidad del 18,47 %, Sharpe 1,529, drawdown −17,83 %, Expected Shortfall diario al 5 % de −2,49 % y exposición media del 95,62 %. Buy & Hold obtuvo 31,97 %, 20,20 %, 1,474, −22,77 %, −2,72 % y 100 %, respectivamente.

V5 conservó el 95,06 % de la rentabilidad anual de Buy & Hold y redujo la magnitud del drawdown un 21,71 %. En comparación con V4, elevó la rentabilidad anual desde 28,71 % hasta 30,39 %, mejoró el Sharpe desde 1,522 hasta 1,529 y redujo ligeramente el drawdown desde −18,29 % hasta −17,83 %. Estos datos satisfacen retrospectivamente el compromiso definido, pero no constituyen validación independiente.

Las políticas no seleccionadas permiten atribuir el comportamiento. `confirmation` conservó una rentabilidad del 31,64 %, con Sharpe 1,513 y drawdown −20,56 %. `minimum_caps` obtuvo 28,82 %, 1,535 y −17,40 %. La máquina de estados asimétrica redujo el drawdown hasta −14,18 % y elevó el Sharpe a 1,562, pero sacrificó rentabilidad hasta 24,94 % y redujo la exposición al 77,73 %. El resultado muestra que una reducción adicional de riesgo es posible, aunque no gratuitamente.

## Diagnóstico de modelos

El objetivo normalizado mantuvo capacidad discriminante moderada. El AUC anual del horizonte de cinco sesiones se situó aproximadamente entre 0,552 y 0,596; el de diez sesiones entre 0,544 y 0,616; y el de veinte sesiones entre 0,547 y 0,667. Algunos horizontes se abstuvieron cuando no alcanzaron 0,55. La normalización por tasa base corrigió la comparación entre probabilidades con prevalencias distintas, pero ninguna política que utilizase riesgo dominó a la tendencia semanal en selección.

HAR-Ridge fue seleccionado con mayor frecuencia al inicio del periodo y VIX pasó a ser el estimador puntual preferido en años recientes. Histogram gradient boosting no ganó por MAE en ningún ajuste anual. La cobertura del percentil 75 cuantílico disminuyó desde aproximadamente 75 % hasta 56 %; por ello se conserva como diagnóstico y no controla directamente la exposición.

## Robustez

Con cinco políticas como número de ensayos, la probabilidad tipo Deflated Sharpe Ratio fue 95,8 %. Este valor no penaliza todas las decisiones de diseño previas y no debe interpretarse aisladamente. El CSCV sobre las cinco políticas y tres benchmarks estimó un PBO del 92,9 %, por lo que la inestabilidad de ranking continúa siendo elevada.

El bootstrap circular de rentabilidad anual frente a Buy & Hold estimó una diferencia media de −1,71 puntos porcentuales, con intervalo del 95 % entre −5,92 y 1,78 puntos. El incremento de costes desde 0 hasta 30 puntos básicos redujo la rentabilidad anual retrospectiva de 30,55 % a 30,06 % y el Sharpe de 1,536 a 1,516. Un retraso adicional de una sesión mantuvo resultados próximos, aunque empeoró el drawdown a −18,38 %.

En los episodios de 2018 y 2022, la tendencia redujo pérdidas y drawdown, pero no mejoró necesariamente el Sharpe durante el propio episodio. En la recuperación de 2020 sacrificó parte del rebote. Esta asimetría explica el coste de protección observado.

## Decisión

V5 sustituye a V4 como candidata para shadow mode porque mejora retrospectivamente rentabilidad, Sharpe y drawdown con una exposición mayor y una política más simple. No se proclama superioridad estadística. La selección final demuestra además que los agentes de riesgo y volatilidad son útiles como diagnóstico y como políticas alternativas, pero la evidencia disponible no justifica que controlen la señal prospectiva principal.

La configuración, código, protocolo y política seleccionada se protegen mediante hashes. Un flujo programado genera una señal diaria auditable, la conserva como artefacto durante noventa días y no ejecuta operaciones.
