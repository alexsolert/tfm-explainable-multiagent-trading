# Revisión de literatura y decisiones de diseño de V4

## Alcance

Esta revisión estudia qué modificaciones son defendibles para mejorar el framework sin confundir complejidad con capacidad predictiva. La búsqueda se centró en predicción de retornos, trend following, gestión de volatilidad, modelos HAR, machine learning tabular, validación financiera y agentes LLM. V4 permanece separada de V2 y V3 y no consulta datos posteriores al 31 de agosto de 2026.

## Síntesis de la evidencia

La predicción directa del retorno de un único ETF presenta una relación señal/ruido baja. Gu, Kelly y Xiu muestran que árboles y redes neuronales pueden capturar interacciones no lineales en grandes paneles de acciones, pero sus resultados dependen de una dimensión cross-sectional muy superior a la disponible en una única serie de QQQ [1]. Por tanto, la evidencia no justifica entrenar una red profunda o un Transformer sobre aproximadamente cuatro mil observaciones diarias. En V4 se comparan regresión logística y histogram gradient boosting, con regularización, selección temporal y abstención.

El time-series momentum documenta persistencia de retornos a horizontes aproximados de uno a doce meses en múltiples clases de activos [2]. V4 no convierte este resultado en una regla automática de compra o venta: crea un agente de tendencia multi-horizonte y limita su autoridad mediante un cap de exposición. Esta decisión evita que una señal lenta controle por sí sola la cartera.

La volatilidad resulta más predecible que el retorno. El modelo HAR-RV representa la persistencia mediante componentes diarios, semanales y mensuales con una especificación parsimoniosa [3]. Evidencia comparativa reciente no encuentra que los modelos no lineales superen de forma general a los lineales en previsión de volatilidad, y señala que modelos sencillos pueden ofrecer mayor utilidad económica [4]. V4 utiliza por ello un HAR regularizado como agente y conserva un volatility-target histórico como benchmark independiente.

Moreira y Muir encuentran que reducir riesgo cuando la volatilidad es elevada puede mejorar el rendimiento ajustado por riesgo [5]. Bollerslev y coautores interpretan el volatility targeting como estabilización del riesgo, separada de la predicción del retorno [6]. V4 aplica esta idea como dimensionamiento de posición, no como evidencia direccional.

La estructura temporal del VIX contiene información sobre primas de varianza y estados de estrés [7]. V4 incorpora la pendiente aproximada VIX/VIX3M, junto con crédito HYG/LQD y amplitud de ETF, como variables del especialista de riesgo. No se interpretan como causalidad ni se utilizan datos futuros.

La evaluación de muchas estrategias infla el Sharpe observado. El Deflated Sharpe Ratio fue propuesto para corregir selección múltiple y no normalidad [8]. Esta iteración registra 1.660 evaluaciones de configuración, incluidas comprobaciones repetidas de robustez sobre objetivos de riesgo, estimadores de volatilidad y políticas de coordinación. En consecuencia, sus resultados se consideran desarrollo e internal validation, no prueba final.

Los modelos lingüísticos especializados, como FinBERT, mejoran tareas de clasificación de sentimiento financiero [9], pero esto no implica automáticamente retornos negociables. BloombergGPT confirma que los modelos financieros requieren grandes corpus especializados [10]. Frameworks recientes como TradingAgents [11] y StockAgent [12] son relevantes para la arquitectura multiagente, aunque sus resultados no sustituyen una evaluación point-in-time sobre noticias históricas verificables. V4 mantiene al LLM fuera de la ejecución; su futura entrada requerirá un corpus timestamped y una ablación específica.

## Alternativas examinadas

| Alternativa | Decisión | Motivo |
|---|---|---|
| Frecuencia semanal | Conservada como V2/V3 | Menor rotación, pero reacción lenta ante shocks. |
| Frecuencia diaria | Adoptada en V4 | Más observaciones y reacción más rápida; exige purga de etiquetas solapadas. |
| Regresión logística | Adoptada | Benchmark regularizado, calibrable y explicable. |
| Histogram gradient boosting | Adoptado | Captura interacciones no lineales con control de profundidad. |
| HAR-RV regularizado | Adoptado | Modelo parsimonioso y respaldado para volatilidad persistente. |
| GARCH/GJR-GARCH | Benchmark futuro | Adecuado para heterocedasticidad, pero HAR permite integrar horizontes y VIX con menor complejidad operativa. |
| Random Forest | No prioritario en V4 | Ya evaluado en V1/V2; no resolvió la generalización direccional. |
| XGBoost/LightGBM/CatBoost | Diferido | Podrían ampliar el benchmark tabular, pero no corrigen la escasez de señal y añaden tuning. |
| LSTM/Transformer | Rechazado por ahora | Muestra temporal insuficiente y elevado riesgo de overfitting. |
| Reinforcement learning | Rechazado por ahora | La recompensa introduce otra capa de optimización retrospectiva y dificulta la atribución. |
| HMM/Markov switching | Diferido | Puede estudiar regímenes, pero su inestabilidad debe compararse con reglas observables simples. |
| Filtro de Kalman | Diferido | Puede suavizar una tendencia latente, aunque añade supuestos dinámicos que deben superar a medias observables. |
| Conformal prediction | Línea futura | Interesante para abstención por incertidumbre; requiere estudiar cobertura bajo dependencia temporal. |
| Panel amplio de acciones | Línea futura | Es la vía más coherente para modelos complejos, pero cambia la pregunta desde QQQ a selección cross-sectional. |
| FinBERT/LLM | Shadow mode futuro | Requiere noticias históricas fechadas y una prueba incremental separada. |

## Granularidad

La frecuencia diaria se adopta para la decisión y el control de riesgo, no para convertir cada fluctuación en una operación. Los predictores conservan horizontes de 5, 20, 60 y 200 sesiones, de modo que la arquitectura es diaria en actualización y multi-horizonte en información. La banda de rebalanceo, el suavizado y los límites continuos evitan que el aumento de frecuencia se traduzca automáticamente en mayor rotación. Este diseño permite reaccionar antes que V2/V3 ante un shock sin renunciar a señales mensuales y anuales.

La alternativa intradía se descarta. Requeriría microestructura, spreads variables, profundidad y timestamps homogéneos que no están disponibles en el conjunto actual. La alternativa semanal permanece como control histórico: reduce ruido y coste, pero responde con retraso. La comparación adecuada, por tanto, no es «diario siempre mejor», sino si el coordinador diario obtiene una mejora ajustada por costes y riesgo en un periodo no usado para seleccionar parámetros.

## Diseño resultante

V4 contiene cuatro especialistas. El agente de riesgo estima la probabilidad calibrada de que QQQ sufra una caída intraperiodo de al menos el 4 % durante las cinco sesiones siguientes. El agente de volatilidad estima la volatilidad realizada de veinte sesiones mediante HAR. El agente de tendencia combina posición respecto a medias de 50 y 200 sesiones y momentum de 20 y 60 sesiones. El agente direccional intenta predecir el signo a cinco sesiones, pero se abstiene si su AUC temporal no alcanza 0,52.

El coordinador utiliza exposición continua entre 25 % y 100 %. La volatilidad objetivo del 32 % evita reducir exposición durante regímenes ordinarios de QQQ. Una tendencia claramente bajista limita la posición al 70 %, mientras que el especialista de riesgo puede imponer un límite adicional. Las decisiones se suavizan y se aplica una banda mínima de rebalanceo para contener costes.

## Referencias

[1] S. Gu, B. Kelly y D. Xiu, “Empirical Asset Pricing via Machine Learning,” *The Review of Financial Studies*, vol. 33, n.º 5, pp. 2223–2273, 2020, doi: 10.1093/rfs/hhaa009.

[2] T. J. Moskowitz, Y. H. Ooi y L. H. Pedersen, “Time Series Momentum,” *Journal of Financial Economics*, vol. 104, n.º 2, pp. 228–250, 2012, doi: 10.1016/j.jfineco.2011.11.003.

[3] F. Corsi, “A Simple Approximate Long-Memory Model of Realized Volatility,” *Journal of Financial Econometrics*, vol. 7, n.º 2, pp. 174–196, 2009.

[4] M. Leushuis y M. Petkov, “Forecasting realized volatility: Does anything beat linear models?,” *Journal of Empirical Finance*, art. 101524, 2024, doi: 10.1016/j.jempfin.2024.101524.

[5] A. Moreira y T. Muir, “Volatility-Managed Portfolios,” *The Journal of Finance*, vol. 72, n.º 4, pp. 1611–1644, 2017, doi: 10.1111/jofi.12513.

[6] T. Bollerslev, B. Hood, J. Huss y L. H. Pedersen, “Risk Everywhere: Modeling and Managing Volatility,” *The Review of Financial Studies*, vol. 31, n.º 7, pp. 2729–2773, 2018, doi: 10.1093/rfs/hhy041.

[7] T. L. Johnson, “Risk Premia and the VIX Term Structure,” *Journal of Financial and Quantitative Analysis*, vol. 52, n.º 6, pp. 2461–2490, 2017, doi: 10.1017/S0022109017000825.

[8] D. H. Bailey y M. López de Prado, “The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality,” *The Journal of Portfolio Management*, vol. 40, n.º 5, pp. 94–107, 2014.

[9] D. Araci, “FinBERT: Financial Sentiment Analysis with Pre-trained Language Models,” arXiv:1908.10063, 2019.

[10] S. Wu *et al.*, “BloombergGPT: A Large Language Model for Finance,” arXiv:2303.17564, 2023.

[11] Y. Xiao, E. Sun, D. Luo y W. Wang, “TradingAgents: Multi-Agents LLM Financial Trading Framework,” arXiv:2412.20138, 2024.

[12] C. Zhang *et al.*, “When AI Meets Finance: Large Language Model-based Stock Trading in Simulated Real-world Environments,” arXiv:2407.18957, 2024, doi: 10.48550/arXiv.2407.18957.
