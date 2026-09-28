# Resultados de desarrollo e internal validation V4

## Estado del resultado

Este documento registra la V4 diaria tras cerrar su búsqueda sobre 2018–2022. El intervalo 2023–31 de agosto de 2026 se denomina internal validation, no test final, y el periodo desde el 1 de septiembre de 2026 permanece sin consultar. Se ejecutaron 1.660 configuraciones o comprobaciones de robustez durante el desarrollo; por ello se informa explícitamente la penalización por selección múltiple.

## Resultado principal

En internal validation, V4 obtuvo una rentabilidad anualizada del 28,71 %, volatilidad del 17,61 %, Sharpe de 1,522, drawdown máximo del −18,29 % y exposición media del 90,92 %. Buy & Hold obtuvo 31,97 %, 20,20 %, 1,474 y −22,77 %, respectivamente. V4 conservó aproximadamente el 89,8 % de la rentabilidad anual de Buy & Hold, aumentó el Sharpe un 3,2 % y redujo la magnitud del drawdown un 19,6 %.

SMA 50/200 produjo una rentabilidad anualizada del 24,29 %, Sharpe de 1,242 y drawdown del −22,77 %. El benchmark de volatility targeting obtuvo 30,89 %, 1,467 y −22,67 %. Por tanto, V4 presentó el mayor Sharpe y el menor drawdown de los cuatro métodos, pero Buy & Hold y volatility targeting conservaron mayor rentabilidad absoluta. No existe superioridad simultánea en todas las métricas.

En el periodo de selección 2018–2022, V4 obtuvo Sharpe 0,580 y drawdown −29,58 %. Mejoró el Sharpe y el drawdown de Buy & Hold, pero quedó por detrás de SMA 50/200, cuyo Sharpe fue 0,727 y cuyo drawdown fue −28,56 %. Esta diferencia limita cualquier afirmación de dominancia estructural.

## Ablaciones

En internal validation, el agente de riesgo aislado alcanzó Sharpe 1,516 y drawdown −20,99 %, mientras que la tendencia aislada alcanzó Sharpe 1,473 y drawdown −18,21 %. La combinación V4 elevó ligeramente el Sharpe a 1,522, aunque su drawdown fue 0,09 puntos porcentuales peor que el de tendencia aislada. El agente de volatilidad aislado se aproximó a Buy & Hold porque el objetivo del 32 % rara vez restringió la exposición. El agente direccional permaneció inactivo y su ablación coincidió con Buy & Hold. Este último resultado es deliberado: la abstención impidió que un clasificador con AUC temporal insuficiente degradase la cartera.

La auditoría anual confirma la especialización. El AUC temporal del agente de riesgo descendió desde 0,784 en el primer ajuste hasta un mínimo de 0,581, pero permaneció por encima del umbral 0,55 en los nueve entrenamientos. El AUC direccional se situó entre 0,451 y 0,507 y nunca alcanzó 0,52. El coordinador utilizó por ello riesgo en producción experimental y mantuvo dirección exclusivamente como salida de diagnóstico.

## Robustez estadística

El bootstrap circular de veinte sesiones y 5.000 remuestreos estimó una diferencia media de rentabilidad anual de V4 frente a Buy & Hold de −3,43 puntos porcentuales, con intervalo del 95 % entre −7,66 y 0,08 puntos. Frente a volatility targeting, la diferencia media fue −2,31 puntos; frente a SMA 50/200 fue +4,39 puntos, pero el intervalo también incluyó cero. Estos cálculos evalúan rentabilidad, no Sharpe ni drawdown.

La probabilidad tipo Deflated Sharpe Ratio fue 31,2 % tras penalizar conservadoramente 1.660 evaluaciones; el Sharpe nulo máximo esperado fue 1,780, superior al 1,522 observado. El CSCV sobre la familia final de ocho series estimó una probabilidad de backtest overfitting del 92,9 %. Esta cifra no cubre todas las políticas probadas, pero constituye una advertencia fuerte. En consecuencia, V4 es una mejora de ingeniería y un candidato prospectivo, no evidencia estadística de una ventaja de inversión persistente.

## Decisión

La arquitectura diaria se conserva porque mejora la gestión del riesgo, documenta correctamente la abstención y ofrece una interfaz más informativa. No se realizarán nuevos ajustes con datos anteriores a septiembre de 2026 bajo la etiqueta V4. Una futura V5 deberá declarar un nuevo protocolo, ampliar la evidencia cross-sectional o incorporar datos point-in-time realmente nuevos. El periodo prospectivo es el único mecanismo capaz de reducir la incertidumbre que señalan el DSR y el PBO.
