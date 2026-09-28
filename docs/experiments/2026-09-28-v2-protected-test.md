# Test protegido V2 · 2025–agosto de 2026

## Apertura

El test se abrió después de ejecutar `qqq-agents v2-freeze`. El archivo
`configs/v2.lock.json` registra el hash SHA-256 de la configuración, el protocolo, las variables,
los modelos, la calibración, la asignación y el código de evaluación. Los datos protegidos se
descargaron en un snapshot independiente y no se utilizaron para modificar la especificación.

## Resultados

| Estrategia | Rentabilidad acumulada | Rentabilidad anualizada | Sharpe | Drawdown máximo | Exposición |
|---|---:|---:|---:|---:|---:|
| Multiagente V2 | 28,93 % | 16,20 % | 0,988 | -15,13 % | 89,20 % |
| Buy & Hold | 39,10 % | 21,53 % | 1,051 | -21,34 % | 98,86 % |
| SMA 50/200 | 12,36 % | 7,13 % | 0,460 | -21,34 % | 87,50 % |
| Agente logístico | 37,21 % | 20,56 % | 1,029 | -21,34 % | 93,18 % |

V2 ejecutó doce cambios de exposición, con un turnover total de 7,0. Permaneció al 100 % durante
71 decisiones y al 50 % durante 17; no alcanzó exposición cero. Con costes de 25 puntos básicos,
la rentabilidad anualizada fue 15,48 % y el Sharpe 0,950. Con 50 puntos básicos, fueron 14,30 % y
0,888 respectivamente.

## Calidad probabilística

La AUC agregada del agente de riesgo fue 0,621 y el Brier score 0,118. La frecuencia observada del
evento adverso fue 13,79 %, frente a una probabilidad media estimada de 16,77 %. El error esperado
de calibración fue 0,030. Estos valores mejoran de forma material la escala de probabilidad de V1.

La AUC direccional fue 0,484. Por tanto, la capa direccional no mostró capacidad de ranking fuera
de muestra; el resultado de V2 procede esencialmente de la exposición estructural y la reducción
gradual asociada al riesgo. La ablación `risk_only` reproduce exactamente la estrategia completa,
mientras que `directional_only` coincide con la posición estructural larga.

## Interpretación

V2 supera con claridad al champion SMA congelado durante el desarrollo y reduce el drawdown frente
a Buy & Hold en aproximadamente 6,2 puntos porcentuales. Sin embargo, no supera a Buy & Hold en
rentabilidad ni Sharpe. El bootstrap por bloques asigna una probabilidad de solo 18,41 % a que V2
supere la rentabilidad anualizada de Buy & Hold; el intervalo del 95 % de la diferencia incluye cero.

El criterio de promoción se evaluó exclusivamente durante desarrollo y seleccionó SMA como champion.
Ese registro no se cambia después del test. El resultado protegido indica que el challenger habría
superado a SMA durante este periodo, pero no autoriza una selección retrospectiva. Para una futura
V3, V2 puede utilizarse como nuevo benchmark y deberá reservarse un periodo posterior o emplearse
paper trading prospectivo.

## Conclusión

La corrección principal funcionó parcialmente: V2 ya no pierde una fase alcista completa y ofrece
una reducción de drawdown económicamente relevante. La hipótesis direccional no queda respaldada;
la aportación demostrada corresponde al agente de riesgo calibrado y a la política long-biased. El
LLM continúa fuera de la comparación predictiva hasta disponer de noticias históricas verificables.
