# V8 final

V8 presenta tres perfiles de exposición a QQQ y una salida actual compatible con paper trading. Los resultados de 2004–2022 se utilizan para selección; 2023–agosto de 2026 es una evaluación retrospectiva, no un holdout.

El perfil conservador prioriza Sharpe y drawdown; el equilibrado busca una combinación intermedia; el agresivo prioriza rentabilidad. Se incluyen benchmarks de exposición constante y exposición media equivalente para evitar atribuir al coordinador la ventaja mecánica del apalancamiento.

El paquete se regenera mediante:

```bash
PYTHONPATH=src .venv/bin/python scripts/export_v8_final.py
```

`current_decision.json` es una salida académica en modo paper y no ejecuta operaciones.
