# Datos y procedencia

## Archivos incluidos

- `source/indpensim_process_original.csv.gz`: las primeras 37 posiciones del CSV `100_Batches_IndPenSim_V3.csv`, con 113.935 filas. Se conserva la precisión numérica leída, los ausentes y los encabezados originales; no es una copia binaria del CSV completo.
- `source/100_Batches_IndPenSim_Statistics.csv`: copia de la tabla auxiliar original. Se utiliza para verificar los indicadores de lote con falla, no para modelar producción en kg.
- `source/provenance.json`: autoría, DOI, licencia, tamaño y SHA-256 del archivo completo del que se extrajo la tabla de proceso.
- `processed/proceso_preparado.csv.gz`: copia con nombres documentados, tipos de metadatos, estrategia de control, temperatura en °C y partición asignada. Se genera al ejecutar el notebook.

Fuente: [Mendeley Data, versión 2](https://data.mendeley.com/datasets/pdnjz7zz5x/2), Stephen Goldrick. Licencia **CC BY 4.0**. Las observaciones son simuladas.

## Corrección de encabezados

El archivo presenta 2.239 nombres en su encabezado y 2.237 valores en las filas examinadas. En la carga completa, dos columnas finales resultan vacías. El bloque usado por este proyecto se identifica por posición, contando desde cero:

| Posición | Nombre utilizado | Evidencia |
|---|---|---|
| 0–30 | Variables del proceso | Nombres y rangos originales; el objetivo está en la posición 13 |
| 31 | `falla_activa` | Indicador binario dentro de los últimos 10 lotes |
| 32 | `control_operador` | 0/1; activo en los lotes 31–60 |
| 33 | `modo_pat` | 1/2; valor 2 en los lotes 61–90 |
| 34 | `lote_repetido` | Secuencia de identificadores 1–100, igual a la posición 35 |
| 35 | `lote` | Coincide con 100 bloques temporales que comienzan en 0,2 h |
| 36 | `lote_con_falla` | Coincide con las 100 etiquetas de la tabla auxiliar |

El [notebook del autor](https://github.com/StephenGoldie/indpensim-notebook) también aplica una corrección de nombres para el identificador. No se usan literalmente `Batch ID` y `Fault flag`, ubicadas después de este bloque, porque contienen señales numéricas incompatibles con esos nombres.

El archivo original se conserva en la ubicación de descarga del usuario. Para regenerar el extracto, descargar y descomprimir la fuente y ejecutar:

```bash
python scripts/extract_process.py /ruta/a/Mendeley_data
```

El script verifica las dimensiones esperadas y no modifica el CSV original. Si cambia la versión, debe revisarse el esquema antes de continuar.

## Ausencias y unidades

Los cinco análisis offline contienen 2.062 valores cada uno; sus vacíos no se imputan ni se utilizan como predictores. Las ausencias de los cambios a una hora se generan al comienzo de cada lote y se imputan dentro de cada ajuste de entrenamiento.

La concentración objetivo se interpreta en g/L y la temperatura se convierte de K a °C. Las señales de gases y aireación conservan la escala publicada; no se usan para balances de masa ni se convierten a porcentajes o volúmenes normalizados. El caudal de extracción conserva su signo negativo.

Los 10 lotes con fallas se conservan para trazabilidad y se marcan como excluidos. Solo los 90 lotes normales participan en la regresión.
