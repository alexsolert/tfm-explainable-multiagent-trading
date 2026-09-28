# Evaluación retrospectiva de V6

## Selección 2004–2022

La selección automática escogió `trend_120`: exposición del 120 % cuando al menos dos de las cuatro señales de tendencia son positivas y del 70 % en caso contrario. V6 obtuvo una rentabilidad anualizada del 12,03 %, Sharpe 0,638 y drawdown máximo del −50,51 %. Buy & Hold obtuvo 11,94 %, 0,629 y −53,40 %, respectivamente. Las diferencias son pequeñas, pero la política cumplió las tres restricciones declaradas antes de examinar el intervalo retrospectivo: mayor rentabilidad, mayor Sharpe y menor magnitud del drawdown.

La exposición constante del 125 % alcanzó una rentabilidad mayor, 13,53 %, pero redujo el Sharpe a 0,604 y aumentó el drawdown hasta −62,70 %. El resultado de V6 no se explica, por tanto, únicamente por aplicar apalancamiento constante.

## Evaluación retrospectiva 2023–agosto de 2026

`trend_120` obtuvo una rentabilidad anualizada del 34,85 %, volatilidad del 21,57 %, Sharpe 1,494 y drawdown máximo del −18,86 %. Buy & Hold obtuvo 31,97 %, 20,20 %, 1,474 y −22,77 %. V6 elevó la rentabilidad anual en 2,88 puntos porcentuales, mejoró el Sharpe en 0,020 y redujo la magnitud del drawdown en 3,91 puntos. La exposición media fue del 113,14 % y permaneció por encima del 100 % durante 792 de 918 sesiones.

El benchmark constante del 125 % produjo 38,49 % anual, pero su Sharpe fue 1,416 y su drawdown −27,91 %. V6 no maximiza la rentabilidad bruta; intenta obtener un compromiso más favorable que Buy & Hold y que el apalancamiento constante.

## Atribución de agentes

La familia `joint`, que exige confirmación de retorno, riesgo y volatilidad, no superó a Buy & Hold. Su mejor variante se aproximó al 13,3 % anual en el periodo completo, con exposición media inferior al 95 % y rotación elevada. Los modelos de retorno tampoco mostraron una señal estable: la mediana de la correlación temporal fue −0,011 a cinco sesiones, −0,013 a diez y 0,030 a veinte. La correlación fue positiva en 8, 10 y 18 de las 23 recalibraciones anuales, respectivamente.

Estos resultados atribuyen la mejora de V6 al agente de tendencia y al dimensionamiento de exposición, no al nuevo agente de retorno. Mantener este resultado negativo es relevante: añadir un modelo predictivo más complejo no mejoró la política y la selección favoreció la ablación parsimoniosa.

## Robustez

Con treinta puntos básicos por cambio de exposición, la rentabilidad retrospectiva permaneció en torno al 34,30 % anual. Al elevar el spread de financiación desde 150 hasta 400 puntos básicos, la rentabilidad descendió desde 34,85 % hasta 34,28 % y el Sharpe desde 1,494 hasta 1,475; el drawdown se mantuvo próximo al −18,9 %. Un día adicional de retraso produjo 34,87 %, Sharpe 1,494 y drawdown −19,78 %.

El bootstrap circular estimó una diferencia media de rentabilidad anual frente a Buy & Hold de 2,95 puntos porcentuales, con intervalo del 95 % entre −2,07 y 8,33 puntos y una frecuencia de superioridad del 87 %. El intervalo incluye cero, por lo que no se acredita superioridad estadística. La probabilidad Deflated Sharpe fue 89,8 % al penalizar diez políticas. CSCV estimó un PBO del 71,4 %, indicador de inestabilidad elevada en el ranking de candidatos.

Durante la crisis financiera global, V6 redujo el drawdown desde −53,40 % hasta −50,51 %. En las ventanas acotadas de COVID-19 y 2022 no mejoró a Buy & Hold: obtuvo drawdowns de −29,68 % y −34,88 %, frente a −28,56 % y −34,83 %. La protección no es uniforme entre crisis.

## Conclusión

V6 constituye la primera versión que supera retrospectivamente a Buy & Hold de forma simultánea en rentabilidad anualizada, Sharpe y drawdown, tanto en el intervalo de selección 2004–2022 como en 2023–agosto de 2026. También supera a V5 en rentabilidad retrospectiva, aunque V5 mantiene mejor Sharpe y menor drawdown. No existe dominancia entre ambas versiones: V5 es preferible para un perfil más conservador y V6 para un objetivo de mayor crecimiento con exposición acotada.

La amplitud del bootstrap, el PBO elevado y el uso previo del periodo retrospectivo impiden formular una conclusión de superioridad persistente. El resultado defendible es que una regla de tendencia con exposición máxima del 120 % resolvió retrospectivamente la limitación de rentabilidad de V5, mientras que el agente ML de retorno no añadió valor demostrable.
