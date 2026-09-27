# Decision 0003: piloto LLM acotado y reproducible

- Estado: aceptada para validación
- Fecha: 2026-09-27

La primera integración LLM se limita a una sola decisión del periodo 2020-2022 y a tres roles:
contexto de mercado, sentimiento y validación estratégica. Cada rol recibe un paquete fechado,
devuelve una salida validada contra un esquema cerrado y no conserva memoria entre fechas. Las
respuestas se almacenan en caché junto con la versión del prompt, el modelo y el uso de tokens.

El coordinador determinista conserva la autoridad sobre la acción final. Las salidas LLM se
transforman al mismo contrato de señal que los agentes cuantitativos y se agregan con los pesos
fijados en `configs/base.yaml`. El perfil de personalidad también se aplica de forma determinista.

Antes de cualquier llamada real se ejecuta una simulación local sin coste. El piloto real requiere
una clave local no versionada, limita la longitud de cada respuesta, estima el coste incremental y
bloquea nuevas llamadas cuando la reserva calculada superaría el presupuesto configurado. El
periodo 2023-2024 permanece excluido. La incorporación de noticias históricas se pospone hasta
disponer de una fuente con fecha y hora de publicación verificables; mientras tanto, el agente de
sentimiento debe responder de forma neutral y declarar la ausencia de evidencia textual.
