# Protocolo experimental V6 de exposición gestionada

## Hipótesis

V6 estudia si una política de exposición acotada entre el 25 % y el 125 % de QQQ puede superar la rentabilidad de Buy & Hold sin empeorar su Sharpe ni su drawdown máximo. Esta hipótesis responde a una limitación estructural de V4 y V5: una política long-only limitada al 100 % puede evitar parte de las caídas, pero no dispone de un mecanismo para recuperar la rentabilidad cedida durante los periodos fuera del mercado.

V6 no modifica ni reinterpreta los resultados congelados de V5. El intervalo 2023–agosto de 2026 ya fue observado durante el desarrollo anterior y se denomina evaluación retrospectiva, no holdout ni prueba final.

## Datos y frecuencia

El histórico principal se amplía desde 2010 hasta el 10 de marzo de 1999, fecha próxima al inicio de QQQ. Se emplean QQQ, SPY, VIX y el rendimiento de Treasury bills a tres meses. La muestra de evaluación comienza en 2004 para disponer de al menos 750 observaciones de entrenamiento y de indicadores de hasta 252 sesiones. La selección de política utiliza exclusivamente 2004–2022; 2023–agosto de 2026 se informa por separado.

La actualización es diaria. El riesgo, la volatilidad y el retorno se estiman a horizontes de 5, 10 y 20 sesiones. La tendencia estructural se observa semanalmente mediante distancia a las medias de 50 y 200 sesiones y momentum de 20 y 60 sesiones. La decisión tomada al cierre solo se aplica al retorno de la sesión posterior.

## Agentes

El agente de riesgo conserva etiquetas de caída normalizadas por la volatilidad. Para cada horizonte, una regresión logística y un histogram gradient boosting compiten mediante predicciones temporales purgadas. Un modelo solo participa si su AUC temporal alcanza 0,53.

El agente de retorno estima el retorno futuro dividido por la volatilidad ex ante del horizonte. Compara Ridge regularizado e histogram gradient boosting con pérdida absoluta. Su error y correlación temporal se registran, aunque su salida no recibe autoridad automática sobre el apalancamiento.

El agente de volatilidad compara HAR-Ridge, histogram gradient boosting, volatilidad realizada rezagada y VIX. El agente de tendencia genera una puntuación observable entre cero y uno. Esta separación permite atribuir si el resultado procede de predicción estadística o de una regla estructural.

## Políticas comparadas

Se comparan tres familias y cinco límites máximos de exposición: 100 %, 110 %, 115 %, 120 % y 125 %.

La familia `joint` exige confirmación conjunta de retorno, tendencia y riesgo, dimensiona la posición por volatilidad y aplica reentrada gradual. La familia `trend` utiliza una regla deliberadamente parsimoniosa: asigna la exposición máxima cuando al menos la mitad de las cuatro señales de tendencia son positivas y limita la posición al 70 % en caso contrario. La familia `guarded_trend` añade un único veto: si la previsión de volatilidad anualizada supera el 40 %, la exposición queda limitada al 70 % independientemente de la tendencia. Este umbral se selecciona sobre 2004–2022 y representa un régimen excepcional, no volatility targeting continuo. Las dos familias de tendencia funcionan también como ablaciones del agente de retorno.

El backtest deduce diez puntos básicos por cambio unitario de exposición. La fracción superior al 100 % paga el rendimiento del efectivo más un spread anual de 150 puntos básicos. La política seleccionada debe superar a Buy & Hold en rentabilidad anual y Sharpe y mantener un drawdown no mayor durante 2004–2022. Si varias políticas cumplen, V6.2 aplica una puntuación predefinida que penaliza drawdown y rotación y añade el peor diferencial de rentabilidad, Sharpe y drawdown observado en tres bloques cronológicos: 2004–2009, 2010–2016 y 2017–2022. El conjunto contiene quince políticas y este número se incorpora a la penalización del Deflated Sharpe Ratio.

## Robustez

La evaluación incluye un benchmark de exposición constante del 125 %, costes entre 0 y 30 puntos básicos, spreads de financiación entre 50 y 400 puntos básicos, un día adicional de retraso, crisis financiera global, COVID-19 y 2022. También se calculan bootstrap circular, Deflated Sharpe Ratio y CSCV/PBO. Estos diagnósticos no convierten la evaluación retrospectiva en evidencia prospectiva.

## Interpretación

La exposición superior al 100 % cambia el perfil económico de la estrategia. Una mejora frente a Buy & Hold debe atribuirse conjuntamente al timing y al presupuesto de riesgo adicional; por ello se informa el benchmark constante del 125 %. V6 se considerará una candidata mejor que V5 únicamente si la selección previa a 2023 y la evaluación retrospectiva muestran una mejora coherente después de costes y financiación.

## Referencias

[1] T. J. Moskowitz, Y. H. Ooi y L. H. Pedersen, “Time Series Momentum,” *Journal of Financial Economics*, vol. 104, n.º 2, pp. 228–250, 2012, doi: 10.1016/j.jfineco.2011.11.003.

[2] A. Moreira y T. Muir, “Volatility-Managed Portfolios,” *The Journal of Finance*, vol. 72, n.º 4, pp. 1611–1644, 2017, doi: 10.1111/jofi.12513.

[3] D. H. Bailey y M. López de Prado, “The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality,” *The Journal of Portfolio Management*, vol. 40, n.º 5, pp. 94–107, 2014.
