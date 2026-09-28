# Evaluación retrospectiva de V6.1

## Selección 2004–2022

La selección automática escogió `guarded_trend_115`: exposición del 115 % cuando al menos dos de las cuatro señales de tendencia son positivas, del 70 % en caso contrario y un veto al 70 % cuando la previsión de volatilidad supera el 40 % anualizado. V6.1 obtuvo una rentabilidad anualizada del 12,12 %, Sharpe 0,667 y drawdown máximo del −45,50 %. Buy & Hold obtuvo 11,94 %, 0,629 y −53,40 %, respectivamente. La política cumplió las tres restricciones declaradas: mayor rentabilidad, mayor Sharpe y menor magnitud del drawdown, con una exposición máxima inferior a la seleccionada por V6.0.

La exposición constante del 125 % alcanzó una rentabilidad mayor, 13,53 %, pero redujo el Sharpe a 0,604 y aumentó el drawdown hasta −62,70 %. El resultado de V6 no se explica, por tanto, únicamente por aplicar apalancamiento constante.

## Evaluación retrospectiva 2023–agosto de 2026

`guarded_trend_115` obtuvo una rentabilidad anualizada del 33,77 %, volatilidad del 20,80 %, Sharpe 1,504 y drawdown máximo del −18,60 %. Buy & Hold obtuvo 31,97 %, 20,20 %, 1,474 y −22,77 %. V6.1 elevó la rentabilidad anual en 1,81 puntos porcentuales, mejoró el Sharpe en 0,029 y redujo la magnitud del drawdown en 4,16 puntos. La exposición media fue del 108,82 % y permaneció por encima del 100 % durante 792 de 918 sesiones.

El benchmark constante del 125 % produjo 38,49 % anual, pero su Sharpe fue 1,416 y su drawdown −27,91 %. V6 no maximiza la rentabilidad bruta; intenta obtener un compromiso más favorable que Buy & Hold y que el apalancamiento constante.

## Atribución de agentes

La familia `joint`, que exige confirmación de retorno, riesgo y volatilidad, no superó a Buy & Hold. Su mejor variante se aproximó al 13,3 % anual en el periodo completo, con exposición media inferior al 95 % y rotación elevada. Los modelos de retorno tampoco mostraron una señal estable: la mediana de la correlación temporal fue −0,011 a cinco sesiones, −0,013 a diez y 0,030 a veinte. La correlación fue positiva en 8, 10 y 18 de las 23 recalibraciones anuales, respectivamente.

Estos resultados atribuyen la mejora de V6 al agente de tendencia y al dimensionamiento de exposición, no al nuevo agente de retorno. Mantener este resultado negativo es relevante: añadir un modelo predictivo más complejo no mejoró la política y la selección favoreció la ablación parsimoniosa.

## Robustez

Al elevar el spread de financiación desde 150 hasta 400 puntos básicos, la rentabilidad descendió desde 33,77 % hasta 33,35 % y el Sharpe desde 1,504 hasta 1,488; el drawdown se mantuvo próximo al −18,6 %. Un día adicional de retraso produjo 33,79 %, Sharpe 1,503 y drawdown −19,43 %.

El bootstrap circular estimó una diferencia media de rentabilidad anual frente a Buy & Hold de 1,82 puntos porcentuales, con intervalo del 95 % entre −2,63 y 6,24 puntos y una frecuencia de superioridad del 79,7 %. El intervalo incluye cero, por lo que no se acredita superioridad estadística. La probabilidad Deflated Sharpe fue 86,3 % al penalizar quince políticas. CSCV mantuvo un PBO del 71,4 %, indicador de inestabilidad elevada en el ranking de candidatos.

Durante la crisis financiera global, V6.1 redujo el drawdown desde −53,40 % hasta −45,50 %. En la ventana de COVID-19 lo redujo desde −28,56 % hasta −24,69 % y elevó el Sharpe desde 0,875 hasta 1,014. En 2022 redujo ligeramente el drawdown desde −34,83 % hasta −33,90 %, aunque su Sharpe fue peor. La protección mejora en las tres ventanas, pero no domina todas las métricas de cada episodio.

## Conclusión

V6.1 supera retrospectivamente a Buy & Hold de forma simultánea en rentabilidad anualizada, Sharpe y drawdown, tanto en el intervalo de selección 2004–2022 como en 2023–agosto de 2026. Frente a V6.0 sacrifica aproximadamente 1,08 puntos de rentabilidad anual retrospectiva, pero reduce exposición, volatilidad y drawdown, y mejora el Sharpe. También supera a V5 en rentabilidad retrospectiva, aunque V5 mantiene mejor Sharpe y menor drawdown. No existe dominancia entre ambas versiones: V5 es preferible para un perfil conservador y V6.1 para un objetivo de mayor crecimiento con exposición acotada.

La amplitud del bootstrap, el PBO elevado y el uso previo del periodo retrospectivo impiden formular una conclusión de superioridad persistente. El resultado defendible es que una regla de tendencia con exposición máxima del 115 % y un veto excepcional de volatilidad resolvió retrospectivamente la limitación de rentabilidad de V5, mientras que el agente ML de retorno no añadió valor demostrable.
