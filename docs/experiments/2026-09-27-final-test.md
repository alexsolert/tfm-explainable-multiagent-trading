# Test final fuera de muestra 2023-2024

## Protocolo

La configuración cuantitativa e híbrida se congeló mediante la decisión 0005 antes de consultar
el periodo final. El test se abrió una sola vez, sin optimizar posteriormente pesos, umbrales,
modelos o personalidad. Se evaluaron 105 decisiones semanales entre el 1 de enero de 2023 y el 31
de diciembre de 2024, con un coste de transacción de 10 puntos básicos y exposición long-only.

## Resultados

| Estrategia | Rentabilidad acumulada | Rentabilidad anualizada | Volatilidad | Sharpe | Drawdown máximo | Exposición |
|---|---:|---:|---:|---:|---:|---:|
| Multiagente híbrido | 23,12 % | 10,85 % | 12,39 % | 0,894 | -15,71 % | 54,29 % |
| Multiagente cuantitativo | 23,12 % | 10,85 % | 12,39 % | 0,894 | -15,71 % | 54,29 % |
| Regresión logística | 36,24 % | 16,55 % | 13,25 % | 1,223 | -8,93 % | 48,57 % |
| Medias 50/200 | 79,29 % | 33,53 % | 16,85 % | 1,804 | -9,82 % | 90,48 % |
| Buy & hold | 92,46 % | 38,30 % | 18,02 % | 1,894 | -9,82 % | 99,05 % |

El sistema multiagente realizó cinco cambios de posición. Redujo la volatilidad respecto a buy &
hold y al cruce de medias, pero obtuvo menor rentabilidad, menor Sharpe y mayor drawdown máximo que
los tres baselines. Este resultado se conserva sin reinterpretar el criterio de éxito: el MVP
demuestra la ejecución y trazabilidad del framework, pero no aporta evidencia de superioridad
financiera en este periodo alcista.

## Componente LLM y control de calidad

El comité híbrido produjo 12 recomendaciones `BUY`, 88 `HOLD` y 5 `SELL`. Solo modificó una
recomendación cuantitativa: el 12 de enero de 2024 transformó `BUY` en `HOLD` cuando la cartera ya
estaba invertida. Por ello, las posiciones y métricas permanecieron idénticas. Las 210 llamadas
pagadas válidas —contexto de mercado y validación estratégica— tuvieron un coste estimado de
`0,401023 USD`; el agente de sentimiento resolvió localmente las 105 fechas sin noticias y sin
consumo de tokens.

La auditoría registró 2 858 referencias de evidencia, ninguna fuera de los paquetes fechados, y
ninguna justificación vacía. Una segunda ejecución recuperó las 315 respuestas —incluido el agente
de sentimiento local— desde la caché, reprodujo las decisiones y tuvo coste incremental nulo. El
coste estimado acumulado de piloto, validación y test final fue aproximadamente `1,011 USD`, por
debajo del límite autorizado de `5 USD`.
