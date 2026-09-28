# Evidencia externa y decisión de diseño de V7

La afirmación de que existen estrategias que superan a Buy & Hold sobre Nasdaq-100 es correcta, pero requiere distinguir tres problemas económicos distintos. Una estrategia puede operar únicamente QQQ, seleccionar acciones individuales pertenecientes al Nasdaq-100 o aumentar el presupuesto de riesgo mediante apalancamiento. Comparar sus rentabilidades sin separar estas fuentes atribuiría al modelo una ventaja que puede proceder de una exposición media superior al mercado.

La evidencia más próxima al alcance del TFM es el trabajo de Gayed y Bilello, que combina medias móviles y apalancamiento condicionado al régimen de tendencia. Los autores sostienen que la exposición apalancada resulta más favorable cuando el mercado está por encima de su media y presenta menor volatilidad, y que debe reducirse cuando ocurre lo contrario [1]. Este enfoque motiva V7 porque conserva QQQ como activo subyacente y modifica únicamente su exposición. No se adopta su resultado como prueba: se reproduce la hipótesis con costes, financiación, señal retrasada y controles de exposición constante.

Moreira y Muir muestran que reducir riesgo cuando la volatilidad realizada aumenta puede mejorar el Sharpe de distintas carteras [2]. No obstante, una evaluación posterior sobre 103 estrategias no encuentra una ventaja sistemática de volatility scaling: mejora el Sharpe en 53 casos y lo empeora en 50, con diferencias significativas en pocos casos [3]. Por ello, V7 incluye volatility targeting como una familia candidata y no como una mejora asumida.

El framework de Cao combina tendencia, trading condicional, stop-loss, volumen y una red MLP para seleccionar parámetros. El resumen publicado informa de mayor rentabilidad y menor drawdown que Buy & Hold para SPY y QQQ entre 2007 y 2023 [4]. La descripción disponible no basta para reproducir exactamente la selección ni cuantificar el riesgo de data snooping; su contribución se utiliza como justificación para explorar interacciones entre tendencia y volatilidad, no para copiar cifras.

RAMP comunica un 24,9 % anual para una estrategia dual-momentum sobre el universo Nasdaq-100 entre 1994 y 2025, frente al 13,9 % del benchmark, y un drawdown inferior al añadir dimensionamiento ATR [5]. El resultado no es directamente comparable: RAMP selecciona acciones individuales mediante momentum relativo y utiliza una base histórica de constituyentes sin survivorship bias. Replicarlo correctamente exigiría datos point-in-time de composición del índice, precios de empresas retiradas y costes por valor. Usar la lista actual de componentes introduciría look-ahead bias. Esta familia queda como ampliación posterior, no como sustituto inmediato de V7.

La cautela es necesaria porque no toda búsqueda técnica encuentra ventaja. Cohen y Cabiri evaluaron osciladores sobre varios ETF y no hallaron ninguna configuración que superase Buy & Hold para QQQ [6]. Además, Nasdaq informa de que menos del 5 % de 193 fondos de gran capitalización growth superó a QQQ durante diez años [7]. QQQ es, por tanto, un benchmark exigente y la existencia de backtests ganadores en Internet no implica que sean comparables, reproducibles o libres de sesgo.

V7 adopta dos perfiles predeclarados. `robust_growth` combina media de 250 sesiones, actualización semanal, volatility target del 35 %, exposición entre 70 % y 150 % y reducción al 70 % bajo tendencia desfavorable. `high_growth` utiliza la misma tendencia, asigna 175 % en régimen favorable y 70 % cuando la volatilidad supera el 35 % o la tendencia se deteriora. Ambas decisiones se aplican una sesión después, pagan diez puntos básicos por cambio unitario y financian la fracción superior al 100 % al tipo libre de riesgo más 150 puntos básicos.

En 2004–2022, reservado para selección, `robust_growth` obtiene un 13,71 % anual, Sharpe 0,651 y drawdown máximo del −50,15 %, frente al 11,93 %, 0,628 y −53,40 % de Buy & Hold. `high_growth` obtiene un 15,03 %, Sharpe 0,640 y drawdown del −53,02 %. En la evaluación retrospectiva 2023–agosto de 2026, los perfiles alcanzan respectivamente 36,95 % y 40,56 % anual, frente a 31,97 % de QQQ. Sin embargo, QQQ conserva mejor Sharpe y drawdown en ese intervalo. V7 amplía la frontera rentabilidad-riesgo; no domina a Buy & Hold en todas las métricas.

Los controles de apalancamiento constante son esenciales. En 2004–2022, una exposición constante del 150 % obtiene 14,79 % anual pero su drawdown alcanza −70,63 %, mientras `robust_growth` sacrifica parte de esa rentabilidad y limita el drawdown a −50,15 %. La contribución defendible de V7 no es que el apalancamiento produzca alpha por sí mismo, sino que el régimen reduce sustancialmente la pérdida extrema frente a un presupuesto de riesgo similar. Para implementación mediante ETF, QLD persigue dos veces el retorno diario del Nasdaq-100, pero ProShares advierte que, a horizontes superiores a un día, la rentabilidad puede desviarse de forma significativa por capitalización y volatilidad [8]. El backtest sintético de V7 modela financiación directa y no debe presentarse como una réplica exacta de QLD.

## Referencias

[1] M. A. Gayed y C. V. Bilello, “Leverage for the Long Run: A Systematic Approach to Managing Risk and Magnifying Returns in Stocks,” 2016 Charles H. Dow Award, rev. 2021, doi: 10.2139/ssrn.2741701.

[2] A. Moreira y T. Muir, “Volatility-Managed Portfolios,” *The Journal of Finance*, vol. 72, n.º 4, pp. 1611–1644, 2017, doi: 10.1111/jofi.12513.

[3] M. Cederburg, M. S. O'Doherty, F. Wang y X. S. Yan, “On the Performance of Volatility-Managed Portfolios,” *Journal of Financial Economics*, vol. 138, n.º 1, pp. 95–117, 2020, doi: 10.1016/j.jfineco.2020.04.015.

[4] B. Cao, “Technical Trading versus Buy and Hold: A Framework Using Common Indicators in the US Stock Market,” *Journal of Investment Strategies*, 2025. [En línea]. Disponible: https://www.risk.net/node/7962623

[5] S. Mirchandani, “RAMP: Dual Momentum for Smarter Investing: A Volatility-Scaled, Drawdown-Controlled Equity Strategy,” SSRN, 2026, doi: 10.2139/ssrn.6980898.

[6] G. Cohen y E. Cabiri, “Algorithmic Setups for Trading Popular U.S. ETFs,” *Cogent Economics & Finance*, vol. 8, n.º 1, 2020, doi: 10.1080/23322039.2020.1720056.

[7] Nasdaq, “The Nasdaq-100 Ecosystem: Institutional Investor Strategies & Implementation,” 2026. [En línea]. Disponible: https://www.nasdaq.com/articles/global-indexes/ndx-ecosystem-institutional-investors

[8] ProShares, “QLD: Ultra QQQ.” [En línea]. Disponible: https://www.proshares.com/our-etfs/leveraged-and-inverse/qld
