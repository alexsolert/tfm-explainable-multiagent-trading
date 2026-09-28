# Protocolo del framework final V8

## Objetivo y alcance

V8 convierte las iteraciones anteriores en un producto experimental único. Su objetivo no es seleccionar retrospectivamente una estrategia ganadora, sino ofrecer tres políticas explicables de exposición a QQQ que correspondan a preferencias de riesgo distintas. El sistema funciona como herramienta de apoyo a la decisión y genera una salida compatible con paper trading; no transmite órdenes ni se presenta como asesoramiento financiero.

La hipótesis evaluada es que la coordinación de agentes especializados puede mejorar diferentes puntos de la frontera histórica entre rentabilidad, volatilidad y drawdown frente a Buy & Hold y frente a exposiciones constantes equivalentes. No se exige que una sola política domine todas las métricas y todos los regímenes.

## Periodos y disciplina temporal

Los datos anteriores a 2004 se utilizan únicamente como warm-up de indicadores. El intervalo 2004–2022 constituye la muestra de selección y se divide adicionalmente en 2004–2009, 2010–2016 y 2017–2022. El periodo 2023–agosto de 2026 se denomina evaluación retrospectiva porque fue consultado durante el desarrollo de V5–V7. Ninguna cifra de este intervalo se etiqueta como holdout prospectivo.

Los agentes actualizan su dictamen al cierre semanal. La exposición decidida se aplica a partir de la sesión posterior. El backtest descuenta diez puntos básicos por cambio unitario de posición y financia la exposición superior al 100 % al rendimiento del efectivo más 150 puntos básicos anuales.

## Agentes y autoridad

El agente de tendencia combina la posición respecto a medias de 50, 200 y 250 sesiones con momentum de 20 y 60 sesiones. Su puntuación determina si el régimen permite exposición normal o requiere una posición defensiva.

El agente de volatilidad emplea la volatilidad realizada a veinte sesiones y clasifica el riesgo como normal, alto o extremo. Es el único agente con autoridad de veto duro: bajo volatilidad extrema limita todos los perfiles. Esta decisión responde a la evidencia de V8 preliminar, en la que conceder el mismo veto al drawdown perjudicaba las recuperaciones y reducía la rentabilidad fuera de muestra temporal.

El agente de drawdown observa la pérdida relativa al máximo de 252 sesiones. Su salida se conserva en la deliberación y en la explicación, pero es consultiva. El agente de fuerza relativa compara QQQ con SPY a veinte sesiones y contextualiza el régimen sin modificar directamente la posición. Esta separación hace visible qué información se consulta y qué agentes poseen autoridad efectiva.

El coordinador produce tres exposiciones simultáneas. El perfil conservador opera entre el 45 % y el 115 %; el equilibrado, entre el 50 % y el 150 %, con volatility target del 35 %; el agresivo, entre el 50 % y el 175 %. En regímenes desfavorables se reducen respectivamente a exposiciones defensivas predefinidas. La selección del perfil corresponde al usuario y no se optimiza según el resultado futuro.

## Benchmarks y robustez

V8 se compara con QQQ Buy & Hold, exposiciones constantes del 115 %, 150 % y 175 %, una regla de media de 250 sesiones, volatility targeting convencional al 35 % y controles constantes con la misma exposición media que cada perfil. Estos controles separan la aportación del timing de la ventaja mecánica derivada del apalancamiento.

La robustez incluye costes y financiación, bloques cronológicos, crisis financiera global, COVID-19 y 2022, bootstrap circular de bloques, Deflated Sharpe Ratio y CSCV/PBO sobre la familia completa. Los intervalos bootstrap y el PBO se muestran incluso cuando son desfavorables. La interpretación se limita a la evidencia disponible: una mejora puntual de la frontera histórica no prueba alpha persistente.

## Criterio de uso futuro

La salida actual contiene fecha, señales de los agentes, puntuación del comité, exposición por perfil, acción respecto a la decisión previa, confianza y explicación. Este contrato permite conectar en el futuro una fuente de datos programada y un adaptador de broker en modo paper. La ejecución con capital real requeriría validación prospectiva, monitorización de desviaciones, límites operativos y revisión regulatoria adicionales.
