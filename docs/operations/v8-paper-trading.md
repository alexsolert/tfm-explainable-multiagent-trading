# Operación de V8 en modo paper trading

## Alcance de seguridad

El motor paper no contiene credenciales de broker, no abre conexiones de ejecución y no transmite órdenes. Las exposiciones superiores al 100 % se representan como notional sintético y como participaciones equivalentes de QQQ. Estas cantidades sirven para simular una cartera, no para copiarse directamente a una cuenta real.

## Primera ejecución

Desde la raíz del repositorio:

```bash
uv run qqq-agents v8-paper --refresh --profile balanced
```

La opción `--refresh` descarga snapshots nuevos de QQQ, SPY, VIX y Treasury bills. El perfil puede ser `conservative`, `balanced` o `aggressive`. El capital inicial predeterminado es 10.000 USD y puede modificarse mediante `--initial-capital` únicamente antes de crear la cuenta.

Para comprobar de nuevo la misma instantánea sin contactar al proveedor:

```bash
uv run qqq-agents v8-paper --profile balanced --through AAAA-MM-DD
```

## Artefactos

El comando mantiene:

- `data/raw/v8_paper/`: snapshots normalizados y manifiesto de validación.
- `artifacts/v8_paper/runs/`: registros inmutables por fecha, cuenta y perfil.
- `artifacts/v8_paper/accounts/`: estado actual de cada cuenta paper.
- `artifacts/v8_paper/ledger.csv`: historial compacto de decisiones y valoración.
- `artifacts/v8_paper/latest.json`: última salida utilizada por el dashboard local.

Cada registro incluye hashes del snapshot, configuración e implementación. Repetir una ejecución con las mismas entradas es idempotente: no duplica el ledger ni vuelve a aplicar costes. Si ya existe un registro para la misma fecha con entradas diferentes, el motor se detiene en lugar de sobrescribirlo.

## Valoración

Entre dos ejecuciones, la exposición anterior se aplica a los retornos diarios observados. La fracción no invertida recibe el rendimiento del efectivo y la fracción superior al 100 % paga el tipo de efectivo más el spread configurado. Al cambiar de exposición se deducen los costes de transacción. La cartera almacena capital, exposición, precio anterior y participaciones equivalentes.

## Validación de datos

La ejecución se detiene si faltan columnas, hay fechas duplicadas, la instantánea contiene datos posteriores a la fecha solicitada, presenta más de siete días naturales de retraso o incluye un movimiento diario de QQQ superior al 50 %. La salida diferencia `market_as_of`, fecha del último precio, y `signal_as_of`, cierre semanal que originó el dictamen.

## Límite de interpretación

El ledger prospectivo puede utilizarse para observar el comportamiento futuro de la configuración congelada, pero una muestra corta no permite inferir superioridad estadística. Cualquier conexión posterior a un broker deberá comenzar en sandbox y añadir límites de notional, autenticación, confirmación humana y reconciliación de posiciones.
