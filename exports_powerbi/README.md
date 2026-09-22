# Tablas para Power BI

Todos los archivos CSV utilizan coma como separador, punto decimal y codificación UTF-8 con BOM. En Power Query, si la configuración regional no interpreta el punto decimal, cambiar el tipo numérico usando una configuración regional compatible. Los valores ausentes permanecen vacíos.

| Tabla | Granularidad | Uso |
|---|---|---|
| `operacion_proceso.csv` | Un lote y una hora transcurrida | Señales operativas y concentración del simulador |
| `lotes_particiones.csv` | Un lote | Estrategia, duración, cantidad de registros y partición |
| `predicciones_prueba.csv` | Un lote y una hora, solo prueba normal | Referencia del simulador, predicción, curva típica y errores |
| `metricas_validacion.csv` | Una configuración | Comparación para selección del modelo |
| `metricas_prueba.csv` | Un modelo | Resultado final reservado |
| `estabilidad_lotes.csv` | Un corte | Comprobación interna de sensibilidad entre lotes |
| `errores_por_lote.csv` | Un lote de prueba | MAE y sesgo por lote |
| `errores_por_tramo.csv` | Un tramo de tiempo | MAE y sesgo según avance del proceso |
| `errores_por_nivel.csv` | Un nivel de concentración | Error en valores bajos, medios y altos |
| `errores_por_estrategia.csv` | Una estrategia | MAE medio entre lotes |
| `sensibilidad_variables.csv` | Un grupo de variables | Aumento del error al desalinear señales |
| `calidad_datos.csv` | Una variable | Ausencias y cantidad de valores distintos |
| `diccionario_variables.csv` | Una posición original | Correspondencia de nombres, significado y uso |

## Relaciones y filtros

Relacionar `lotes_particiones[lote]` con `operacion_proceso[lote]`, `predicciones_prueba[lote]` y `errores_por_lote[lote]` mediante relaciones **uno a varios**, con filtro desde la tabla de lotes. Las tablas agregadas de métricas pueden mantenerse independientes.

La clave de lectura es `lote + tiempo_h`. Para cruzar operación y predicciones, utilizar esa clave compuesta o filtrar mediante la dimensión de lote; no unir únicamente por tiempo, porque todos los lotes reinician en 0,2 h.

Los 10 lotes excluidos aparecen en operación y en la dimensión para trazabilidad. Aplicar el filtro de partición adecuado para no presentarlos como parte de la evaluación. Las predicciones contienen exclusivamente los 18 lotes de prueba normal.

## Páginas sugeridas

1. **Resumen:** MAE medio entre lotes, RMSE, R², mejora frente a curva típica y alcance simulado.
2. **Proceso:** selección de lote, curvas de concentración y señales operativas.
3. **Evaluación:** observado/predicho, error por tramo, estrategia y lote.
4. **Calidad y metodología:** particiones, variables excluidas y procedencia.

La concentración es una magnitud intensiva: no sumar g/L para representar producción. No sumar MAE ni R². Para el indicador principal usar el promedio de los MAE por lote, no el promedio ponderado por cantidad de filas. El tiempo representa horas desde el inicio de un lote, no una fecha de calendario.
