# Protocolo experimental V5 multi-frecuencia

## Objetivo

V5 estudia si una arquitectura multi-frecuencia puede conservar una proporción mayor de la rentabilidad de Buy & Hold y, simultáneamente, reducir su drawdown. La hipótesis no depende de predecir el signo del retorno diario. Riesgo y volatilidad se actualizan diariamente; la tendencia estructural se actualiza semanalmente; el coordinador asigna exposición long-only a QQQ y efectivo remunerado.

V4 permanece congelada. V5 constituye una línea nueva y no puede reinterpretar 2023–agosto de 2026 como holdout, porque ese intervalo ya se observó durante el desarrollo de V4. Se denomina evaluación retrospectiva. El periodo prospectivo de V5 comienza el 29 de septiembre de 2026, después de cerrar su código y configuración.

## Objetivos de riesgo

El evento adverso deja de definirse mediante una caída nominal fija. Para cada horizonte de 5, 10 y 20 sesiones, se calcula el peor retorno acumulado desde la fecha de decisión y se compara con un umbral proporcional a la volatilidad realizada de veinte sesiones:

\[
u_{t,h}=-k_h\,\sigma_{t,20}\sqrt{h/252}.
\]

Los multiplicadores son 1,25, 1,50 y 1,75. Así, un movimiento se clasifica por su severidad relativa al régimen vigente. Como los tres horizontes poseen prevalencias diferentes, cada probabilidad se divide por la tasa base observada en su conjunto de entrenamiento: una intensidad de 1 representa riesgo ordinario y un valor superior a 1 representa riesgo por encima de la base histórica. Las intensidades activas se combinan con pesos 0,50, 0,30 y 0,20. Cada etiqueta solo entra en entrenamiento cuando ha concluido todo su horizonte.

## Laboratorio de volatilidad

Se comparan cuatro estimadores point-in-time: HAR-Ridge, histogram gradient boosting, volatilidad realizada rezagada y VIX. La selección anual utiliza exclusivamente predicciones temporales purgadas y error absoluto medio. Un gradient boosting cuantílico independiente estima el percentil 75 de la volatilidad futura; se registra su cobertura, pero el coordinador base utiliza la previsión puntual para evitar una reducción sistemática excesiva de exposición. HAR se incluye por su representación parsimoniosa de componentes de distinta frecuencia [1], mientras que los modelos lineales permanecen como referencia porque no existe evidencia general de que los métodos no lineales los superen en todos los horizontes [2].

## Políticas de coordinación

Se comparan cinco políticas con las mismas predicciones y costes:

1. `risk_only`: exposición determinada únicamente por el riesgo multi-horizonte.
2. `trend_only`: límite de exposición definido por tendencia semanal.
3. `minimum_caps`: mínimo entre riesgo, tendencia y volatility targeting, equivalente conceptual de V4.
4. `confirmation`: exige coincidencia entre riesgo elevado y tendencia bajista para adoptar el estado más defensivo.
5. `asymmetric_state_machine`: estados NORMAL, ALERTA, DEFENSIVO y RECUPERACIÓN; los recortes son inmediatos y la reentrada está limitada a diez puntos porcentuales diarios.

La selección utiliza únicamente 2018–2022. Se calcula la frontera de Pareto sobre rentabilidad, Sharpe, drawdown y rotación. Se intenta satisfacer tres restricciones: conservar al menos el 95 % de la rentabilidad anual de Buy & Hold, reducir el drawdown un 20 % y no empeorar su Sharpe. Si ninguna política las satisface conjuntamente, se elige dentro de la frontera de Pareto mediante una función predefinida y se declara el incumplimiento.

## Evaluación y sensibilidad

El intervalo 2023–31 de agosto de 2026 es una evaluación retrospectiva, nunca una prueba final. Se presentan todas las políticas, no solo la seleccionada. Se calculan Expected Shortfall al 5 %, Calmar, costes de 0, 5, 10, 20 y 30 puntos básicos, retrasos de ejecución de una y dos sesiones y episodios de estrés de 2018, 2020 y 2022. La incertidumbre se evalúa con bootstrap circular, Deflated Sharpe Ratio y CSCV/PBO [3].

## Criterio de continuidad

V5 solo reemplazaría a V4 como candidata prospectiva si mejora su compromiso entre rentabilidad, drawdown, Sharpe y robustez sin depender de un incremento importante de rotación. Incluso si lo consigue retrospectivamente, la existencia de ventaja persistente solo podrá juzgarse con observaciones posteriores a la congelación.

## Referencias

[1] F. Corsi, “A Simple Approximate Long-Memory Model of Realized Volatility,” *Journal of Financial Econometrics*, vol. 7, n.º 2, pp. 174–196, 2009.

[2] M. Leushuis y M. Petkov, “Forecasting realized volatility: Does anything beat linear models?,” *Journal of Empirical Finance*, art. 101524, 2024, doi: 10.1016/j.jempfin.2024.101524.

[3] D. H. Bailey y M. López de Prado, “The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality,” *The Journal of Portfolio Management*, vol. 40, n.º 5, pp. 94–107, 2014.
