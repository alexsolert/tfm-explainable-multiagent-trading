# Validación híbrida completa 2020-2022

## Ejecución

Se evaluaron 157 fechas mediante dos llamadas pagadas por decisión: contexto de mercado y
validación estratégica. El agente de sentimiento se resolvió mediante la compuerta local de
evidencia. Todas las respuestas quedaron asociadas al snapshot, versión del prompt, tokens y
paquete fechado correspondiente. El coste estimado de las respuestas válidas fue `0,601601 USD`.

## Resultados

La estrategia híbrida obtuvo una rentabilidad acumulada del `45,05 %`, una rentabilidad anualizada
del `13,11 %`, un ratio de Sharpe de `0,8624` y un drawdown máximo del `-12,92 %`. La exposición fue
del `53,50 %` y se produjeron ocho cambios de posición. Estas métricas coinciden con las del sistema
cuantitativo porque la única diferencia de acción se produjo cuando la posición resultante ya era
larga.

## Control de calidad

No hubo justificaciones vacías. Los 157 resultados de sentimiento fueron neutrales y no consumieron
tokens. Una auditoría inicial detectó variantes sintácticas en 68 referencias de evidencia; se
incorporó una normalización determinista contra el paquete de entrada y la auditoría posterior
obtuvo cero referencias inválidas. Una repetición integral recuperó las 471 salidas desde caché con
coste incremental nulo y reprodujo exactamente las métricas.
