# Protocolo experimental V2

## Hipótesis

H1. La calibración temporal reduce la diferencia entre probabilidad de riesgo estimada y
frecuencia observada respecto a V1.

H2. Una posición estructural larga y la reducción gradual de exposición evitan el retraso de
entrada observado en 2023.

H3. Variables de régimen externas aportan información incremental frente a transformaciones
adicionales del precio de QQQ.

H4. Un LLM solo puede aportar valor incremental si recibe evidencia textual histórica y fechada
que no esté contenida en las variables cuantitativas.

## Partición temporal

- 2015–2019: entrenamiento inicial.
- 2020–2024: desarrollo, diagnóstico, selección de modelos y ablaciones.
- 2025-01-01–2026-08-31: test protegido, con apertura única tras congelar la configuración.

El walk-forward se actualiza en cada cambio de año. Las etiquetas cuyo final exceda el cutoff se
excluyen. La selección interna usa TimeSeriesSplit, purga por `target_end_date` y un embargo de una
decisión semanal.

## Variables y modelos

Los grupos técnico, momentum, riesgo y régimen no comparten variables. Para cada agente se comparan
regresión logística regularizada, random forest e histogram gradient boosting. El criterio de
selección es el Brier score obtenido mediante predicciones temporales fuera de muestra; el AUC se
conserva como medida de discriminación. Tras seleccionar el modelo se estima una calibración Platt
sobre sus predicciones out-of-fold y se reentrena el estimador base con toda la historia permitida.

## Política de exposición

La cartera utiliza exposiciones 0, 0,5 y 1. La exposición por defecto es 1. Una probabilidad de
riesgo moderada reduce la exposición a 0,5; una probabilidad severa la reduce a 0. Una evidencia
direccional bajista sin riesgo severo solo puede reducirla a 0,5. Los umbrales quedan fijados en
`configs/v2.yaml` antes de abrir el test protegido.

## Evaluación

Se informan rentabilidad acumulada y anualizada, volatilidad, Sharpe, drawdown máximo, exposición,
turnover y número de operaciones reales. La calidad probabilística se mide con AUC, Brier, error de
calibración y tablas de fiabilidad por año. Se evalúan costes de 10, 25 y 50 puntos básicos.

Las ablaciones predefinidas son: exposición estructural larga, solo dirección, solo riesgo, riesgo
binario calibrado y riesgo sin calibrar. Se calcula bootstrap circular por bloques de cuatro semanas
y CSCV sobre el conjunto de estrategias para estimar la probabilidad de backtest overfitting.

## Criterio de promoción

El challenger V2 únicamente pasa a ser champion si supera el Sharpe de SMA 50/200 y la probabilidad
bootstrap de superar su rentabilidad anualizada es al menos 0,90. Este criterio se aplica antes del
test protegido. Si no se cumple, SMA continúa como champion y V2 se ejecuta en shadow mode.

## LLM y noticias

El proveedor o tamaño del LLM no forma parte de la optimización inicial. Las noticias deberán incluir
`evidence_id`, `published_at`, `text`, `source` y, cuando proceda, `ticker`. El sistema rechaza evidencia
posterior a la decisión. Se compararán cuantitativo, cuantitativo más noticias y comité completo. Una
variante LLM que no modifique posiciones o utilidad no se considerará mejora predictiva.
