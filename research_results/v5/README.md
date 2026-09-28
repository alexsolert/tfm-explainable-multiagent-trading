# Paquete de auditoría V5

Este directorio contiene resultados compactos y versionados; no incluye precios históricos ni credenciales.

- `metrics.json`: métricas, sensibilidad, estrés y diagnósticos estadísticos.
- `training_audit.csv`: cortes temporales, AUC, Brier, prevalencias y errores de volatilidad.
- `model_leaderboard.csv`: candidatos evaluados en cada reentrenamiento anual.
- `policy_leaderboard.csv`: frontera de Pareto y política seleccionada con 2018–2022.

El intervalo 2023–agosto de 2026 es retrospectivo y no debe denominarse holdout. La política prospectiva congelada es `trend_only` y comienza el 29 de septiembre de 2026.
