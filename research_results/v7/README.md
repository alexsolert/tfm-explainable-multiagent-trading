# V7: frontera de crecimiento

Este directorio contiene resultados reproducibles de dos políticas QQQ con exposición de hasta 150 % y 175 %. Los parámetros se seleccionaron exclusivamente con 2004–2022. El periodo 2023–agosto de 2026 ya había sido observado en iteraciones anteriores y se informa como evaluación retrospectiva, no como holdout.

`robust_growth` prioriza la consistencia temporal: supera la rentabilidad de Buy & Hold en los tres bloques de selección y reduce de forma material el drawdown frente a una exposición constante del 150 %. `high_growth` prioriza rentabilidad absoluta y acepta mayor volatilidad. Ninguna política domina a QQQ en todas las métricas durante el periodo retrospectivo.

Los resultados se regeneran con:

```bash
PYTHONPATH=src .venv/bin/python scripts/export_v7_research_results.py
```
