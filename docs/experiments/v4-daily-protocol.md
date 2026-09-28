# Protocolo experimental V4 diario

## Pregunta de investigación

V4 estudia si una coordinación diaria centrada en riesgo, volatilidad y tendencia puede mejorar el rendimiento ajustado por riesgo de una exposición long-only a QQQ sin depender de una predicción direccional débil. No reemplaza los resultados congelados de V1/V2 ni se presenta como test final.

## Separación temporal

Los datos comienzan en 2010. Cada modelo anual se entrena únicamente con filas cuyo objetivo ya ha finalizado antes del 1 de enero correspondiente. El periodo 2018–2022 se utiliza para seleccionar la política; 2023–31 de agosto de 2026 constituye validación interna; desde el 1 de septiembre de 2026 queda reservado para observación prospectiva. El código bloquea su apertura por defecto.

Las etiquetas direccionales y de riesgo utilizan cinco sesiones futuras. El riesgo se define como una caída mínima de al menos el 4 % durante ese horizonte. La volatilidad objetivo es la volatilidad realizada de las veinte sesiones siguientes. Las particiones internas son temporales y aplican una purga de veinte sesiones.

## Agentes y coordinación

El agente de riesgo compara regresión logística e histogram gradient boosting mediante AUC y Brier temporales y se abstiene si el AUC no llega a 0,55. El agente direccional realiza la misma comparación, pero se abstiene por debajo de 0,52. El agente de volatilidad utiliza una regresión Ridge HAR con componentes diaria, semanal y mensual. El agente de tendencia resume precio frente a medias de 50 y 200 sesiones y momentum de 20 y 60 sesiones.

El coordinador no suma recomendaciones discretas. Aplica límites de exposición y selecciona el más conservador: volatility targeting, probabilidad de riesgo, tendencia y, solo si está activo, dirección. La exposición permanece entre 25 % y 100 %, se suaviza y no se modifica si el cambio propuesto es inferior a cinco puntos porcentuales. El efectivo se remunera con el rendimiento histórico aproximado de las letras del Tesoro y cada cambio de exposición soporta diez puntos básicos.

## Comparadores y ablaciones

Los comparadores externos son Buy & Hold, SMA 50/200 y volatility targeting basado exclusivamente en la volatilidad realizada de veinte sesiones. Las ablaciones internas aíslan riesgo, volatilidad, tendencia y dirección. Una ablación no es una técnica externa ni un resultado independiente: elimina componentes para atribuir el comportamiento del framework.

## Control del sobreajuste

Durante el desarrollo se ejecutaron 1.660 configuraciones, incluidas comprobaciones repetidas de robustez. Se exploraron cuatro definiciones de evento de riesgo, cuatro estimadores o combinaciones de volatilidad, objetivos de volatilidad entre 18 % y 32 %, límites bajistas entre 50 % y 100 %, sensibilidades al riesgo y suavizado. La selección utilizó exclusivamente 2018–2022.

La salida informa tres diagnósticos adicionales: bootstrap circular de bloques de veinte sesiones y 5.000 remuestreos contra cada benchmark en validación interna; una probabilidad tipo Deflated Sharpe Ratio penalizada de forma conservadora por las 1.660 evaluaciones; y CSCV/PBO sobre las ocho series finalmente exportadas. Este último diagnóstico no representa las 1.660 políticas y se etiqueta expresamente como análisis limitado de la familia final.

## Criterio de interpretación

El objetivo no es superar cada benchmark en todas las métricas, algo que favorecería la selección retrospectiva. Se valora conjuntamente Sharpe, drawdown, rentabilidad, exposición y estabilidad anual. Una mejora de Sharpe o drawdown que sacrifique demasiada rentabilidad debe declararse. La validación interna no autoriza nuevas modificaciones de parámetros; cualquier cambio posterior crea una nueva versión y convierte este periodo en desarrollo.
