# Especificacion experimental del MVP

## 1. Proposito

El experimento evalua si una arquitectura multiagente jerarquica e hibrida puede aproximarse
al rendimiento ajustado por riesgo de una estrategia buy & hold sobre QQQ, manteniendo una
traza reconstruible de cada decision. El objetivo principal no es maximizar la rentabilidad.

Este documento se congela antes de ejecutar la prueba final. Cualquier cambio posterior debe
registrarse junto con su motivo y no puede justificarse exclusivamente por una mejora obtenida
sobre el periodo de test.

## 2. Datos y calendario

- Activo: Invesco QQQ Trust (`QQQ`).
- Fuente inicial: Yahoo Finance mediante `yfinance`.
- Granularidad: diaria, con precios ajustados.
- Periodo: 1 de enero de 2015 a 31 de diciembre de 2024.
- Frecuencia de decision: semanal, utilizando la ultima sesion disponible de cada semana.
- Ejecucion: la posicion decidida con informacion de la fecha `t` se aplica desde la siguiente
  observacion disponible. Nunca se aplica sobre el retorno que produjo las variables de `t`.

## 3. Particion temporal

- Entrenamiento inicial: 2015-2019.
- Validacion y walk-forward: 2020-2022.
- Prueba final fuera de muestra: 2023-2024.

Los hiperparametros, pesos y umbrales se seleccionan antes de consultar los resultados del
periodo final. El test solo se utiliza para la estimacion final del rendimiento.

## 4. Objetivos predictivos

- Agente tecnico: probabilidad de que el retorno de las cinco sesiones siguientes sea positivo.
- Agente de momentum: probabilidad de continuacion positiva durante las cinco sesiones
  siguientes, usando exclusivamente variables de momentum.
- Agente de riesgo: probabilidad de que el minimo retorno acumulado durante las cinco sesiones
  siguientes sea igual o inferior al -3 %.

Las etiquetas futuras se utilizan para entrenar y evaluar, pero nunca forman parte de la entrada
disponible al agente en la fecha de decision.

## 5. Agentes del MVP

### 5.1 Cuantitativos

- Tecnico: Random Forest sobre tendencia, medias, RSI y MACD, complementado con reglas.
- Momentum: Random Forest sobre retornos y aceleracion a distintos horizontes, complementado
  con reglas.
- Riesgo: Random Forest sobre volatilidad, ATR y drawdown, mas reglas de veto.

Se mantiene la misma familia de modelos para que las diferencias entre agentes procedan de la
informacion y el objetivo, no de mezclar algoritmos incomparables. La regresion logistica actua
como baseline interpretable.

### 5.2 Basados en modelos de lenguaje

- Contexto de mercado.
- Sentimiento.
- Validacion estrategica.

El modelo inicial sera un snapshot fijado de `gpt-5.4-mini`. Las entradas estaran fechadas y
las salidas se validaran contra un esquema estructurado. Se almacenaran en cache para impedir
variaciones y costes repetidos. Laya queda previsto como adaptador experimental de decisiones
cerradas, pero no sustituye la explicacion narrativa exigida a los agentes LLM.

## 6. Coordinacion

Cada agente devuelve una senal en `[-1, 1]`, una confianza en `[0, 1]`, una explicacion y sus
evidencias. El coordinador calcula una contribucion ponderada por peso y confianza. Los pesos
iniciales figuran en `configs/base.yaml` y se ajustan solo durante validacion.

La recomendacion provisional es:

- `BUY` cuando la puntuacion supera `0.20`.
- `SELL` cuando es inferior a `-0.20`.
- `HOLD` en el intervalo intermedio.

El agente de riesgo puede bloquear una compra o forzar una salida si la probabilidad del evento
adverso alcanza `0.65`. El resultado operativo es una exposicion long-only binaria: `1` invertido
y `0` en efectivo.

## 7. Personalidades

Las personalidades son configuraciones transversales, no agentes duplicados. Modifican umbrales,
pesos y prompts. El perfil conservador sera el experimento principal; los perfiles agresivo y
oportunista se analizaran en escenarios y como prueba de sensibilidad.

## 8. Backtesting

- Capital inicial nominal: 10 000 unidades monetarias.
- Coste por cambio de exposicion: 10 puntos basicos por unidad de turnover.
- Sin posiciones cortas, apalancamiento ni costes de financiacion.
- Sin optimizacion de parametros sobre el conjunto final de prueba.

Baselines:

1. Buy & hold.
2. Cruce de medias simples de 50 y 200 sesiones.
3. Regresion logistica como modelo unico.

Metricas: rentabilidad acumulada y anualizada, volatilidad anualizada, ratio de Sharpe sin tasa
libre de riesgo, maximo drawdown, acierto direccional, turnover y numero de cambios de posicion.

## 9. Explicabilidad

- SHAP se aplica sistematicamente a los modelos basados en arboles.
- LIME se aplica a decisiones representativas y discrepantes.
- Los agentes LLM producen una justificacion breve basada solo en la evidencia entregada.
- La traza final conserva entradas, versiones, senales, confianza, pesos, veto y accion.

## 10. Criterios de aceptacion

El MVP se considera completo cuando un tercero puede reconstruir los datos, ejecutar el backtest,
obtener las tablas y abrir el dashboard; ademas, cada decision debe poder explicarse sin inspeccionar
el codigo. Un resultado inferior a buy & hold no invalida el artefacto si la evaluacion es correcta
y las decisiones resultan trazables; debe documentarse sin reinterpretar a posteriori el criterio
de exito.

