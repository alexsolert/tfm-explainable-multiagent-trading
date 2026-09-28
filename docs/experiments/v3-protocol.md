# Protocolo experimental V3: predicción cross-asset y asignación continua

## Estado

V3 es una línea de investigación independiente. V2 permanece congelado y sus resultados no se reescriben. El periodo 2025-01-01–2026-08-31, ya consultado durante V2, se incorpora únicamente al desarrollo de V3. El periodo prospectivo de V3 comienza el 2026-09-01 y no puede utilizarse para seleccionar variables, modelos, umbrales ni hiperparámetros.

## Hipótesis

La hipótesis principal es que un modelo entrenado sobre un panel de activos relacionados puede estimar el retorno condicional de QQQ con mayor estabilidad que los especialistas direccionales entrenados exclusivamente sobre su historia semanal. El diagnóstico inicial descartó el pooling directo para el agente de riesgo, por lo que este se entrena únicamente con observaciones de QQQ y conserva las variables cross-asset como contexto. La hipótesis secundaria es que una asignación continua, remunerando el efectivo y penalizando el riesgo previsto, puede conservar la mayor parte de la rentabilidad estructural de QQQ con un drawdown inferior.

## Datos y objetivos

El panel de desarrollo reúne QQQ, SPY, IWM, SMH y TLT. VIX se utiliza como variable de contexto, no como activo objetivo. Todas las variables son retrospectivas y las etiquetas corresponden al retorno entre decisiones consecutivas. El modelo central estima el retorno futuro; un segundo modelo estima el cuantil 20 % de ese retorno. La rentabilidad del capital no invertido se deriva de la rentabilidad histórica de las letras del Tesoro a 13 semanas.

## Validación

La evaluación utiliza walk-forward anual, ventana de entrenamiento expansiva, separación temporal por fechas, embargo de una decisión y selección entre regresión Ridge e histogram gradient boosting mediante MAE fuera de muestra. Todos los activos de una misma fecha permanecen en el mismo bloque temporal. La estrategia se compara con Buy & Hold, SMA 50/200 y una cartera QQQ con objetivo de volatilidad del 15 %, aplicando los mismos costes y la misma remuneración del efectivo.

## Criterio de éxito prospectivo

El objetivo no es maximizar retrospectivamente una única métrica. Se considera evidencia favorable conservar al menos el 90 % de la rentabilidad de Buy & Hold y reducir su drawdown máximo al menos un 20 %, siempre que el resultado supere también el benchmark con objetivo de volatilidad en Sharpe. Este criterio se evaluará únicamente cuando exista una ventana prospectiva suficiente.

## Diagnóstico inicial de desarrollo

El agente de retorno cross-asset presentó una correlación aproximada de 0,04 con el retorno siguiente de QQQ, por lo que permanece en shadow mode. El pooling directo de activos para predecir riesgo empeoró las métricas incluso después de incorporar la identidad del activo; este diseño se descartó. El especialista de riesgo entrenado solo con QQQ recuperó AUC temporales aproximadas de 0,60–0,66 en la mayor parte del desarrollo. La política seleccionada con 2020–2024 mantiene una exposición estructural del 95 % y solo reduce riesgo a partir de una probabilidad calibrada del 25 %.

En la validación interna 2025–agosto de 2026, V3 obtuvo un Sharpe de 1,045 frente a 1,031 de Buy & Hold y un drawdown de −20,32 % frente a −21,34 %. La mejora es demasiado pequeña para satisfacer el criterio de éxito: V3 permanece como línea prospectiva y no sustituye a V2 ni a los benchmarks.
