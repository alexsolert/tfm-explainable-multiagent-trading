# ADR 0007: V2 calibrada, long-biased y evaluada como challenger

## Estado

Aceptada para desarrollo. El periodo protegido 2025–2026 permanece cerrado hasta que
`configs/v2.yaml` quede bloqueado mediante su hash SHA-256.

## Contexto

V1 obtuvo una rentabilidad acumulada del 23,12 % en 2023–2024 frente al 92,46 % de buy and
hold. El análisis atribuye la diferencia principalmente a una exposición del 15,4 % durante
2023, probabilidades de riesgo sin calibrar y agentes con capacidad predictiva inestable.
El veto fue útil en 2022, pero perjudicial en 2024. El comité LLM no alteró ninguna posición
porque no recibió información exógena.

## Decisión

V1 se conserva sin cambios como benchmark. V2 incorpora:

- grupos de variables no solapados y un agente opcional de régimen;
- comparación anual de regresión logística, random forest e histogram gradient boosting;
- predicciones temporales purgadas y un embargo de una decisión;
- calibración Platt entrenada exclusivamente con predicciones out-of-fold pasadas;
- abstención de agentes cuya AUC temporal no alcanza el mínimo predefinido;
- pesos de agentes direccionales basados en Brier y regularizados hacia pesos iguales;
- posición estructural larga y exposiciones discretas de 0 %, 50 % y 100 %;
- riesgo gradual en lugar de veto jerárquico irreversible;
- etiquetas definidas entre fechas de decisión consecutivas;
- ablaciones, sensibilidad a costes, bootstrap por bloques y estimación CSCV de PBO;
- separación entre dictamen semanal y operación efectivamente ejecutada.

V2 se trata como challenger. Solo puede sustituir al benchmark SMA 50/200 si, durante el
desarrollo, supera su Sharpe y alcanza una probabilidad bootstrap de superarlo igual o superior
al 90 %. Si no satisface ambas condiciones, permanece en shadow mode.

## Consecuencias

La calibración no se interpreta como creación de capacidad predictiva. Un agente bien calibrado
pero sin discriminación debe abstenerse. La incorporación de datos externos se limita inicialmente
a snapshots anteriores a 2025 de SPY, IWM, SMH, TLT y VIX. Las noticias solo podrán entrar mediante
el contrato `HistoricalNewsStore`, con identificador, fuente y timestamp verificables.

Los resultados de 2020–2024 son diagnóstico y desarrollo. No volverán a denominarse test final.
El test protegido es 2025-01-01 a 2026-08-31 y no podrá abrirse si el hash de configuración ha
cambiado después de la congelación.
