# Decision 0004: posponer la fuente de noticias históricas

- Estado: aceptada para el MVP
- Fecha: 2026-09-27

La incorporación de noticias históricas se pospone hasta una iteración posterior al MVP. Una fuente
sin marcas temporales fiables podría introducir información posterior a la decisión y comprometer
la validez del backtesting. La selección, licencia e integración de esa fuente también ampliaría el
alcance previsto para la entrega de una a dos semanas.

Mientras no existan titulares admisibles, el agente de sentimiento devuelve de forma determinista
una señal neutral con confianza nula. Esta regla se aplica antes de invocar el modelo de lenguaje,
por lo que evita coste y no atribuye al modelo una valoración sin evidencia. El rol permanece en la
arquitectura y conserva el mismo contrato para permitir la mejora posterior sin modificar el
coordinador ni los artefactos de trazabilidad.
