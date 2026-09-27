# Decision 0002: calibracion funcional del perfil conservador

- Estado: aceptada para validacion
- Fecha: 2026-09-27

La primera ejecucion walk-forward del perfil conservador inicial produjo una unica compra y 53
vetos entre 2020 y 2022. El sistema permanecia practicamente siempre en efectivo, por lo que la
configuracion no permitia evaluar una estrategia de trading activa ni comparar de manera informativa
la deliberacion de los agentes.

Se moderan los parametros antes de abrir el test final: el multiplicador de senal pasa de 0,85 a
0,95; el suelo de confianza, de 0,60 a 0,52; el incremento del umbral de compra, de 0,10 a 0,03;
el multiplicador del peso de riesgo, de 1,30 a 1,15; y la reduccion del umbral de veto, de -0,10 a
-0,02. El cambio responde al criterio funcional de evitar una estrategia degenerada, no a escoger
la configuracion que maximiza la rentabilidad de validacion.

El periodo 2023-2024 permanece sin consultar.

