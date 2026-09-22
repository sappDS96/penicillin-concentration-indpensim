"""Convierte las interpretaciones calculadas en celdas Markdown y genera una vista HTML."""
from pathlib import Path
import ast
import json
import re
import nbformat
import pandas as pd
from nbconvert import HTMLExporter

root = Path(__file__).resolve().parents[1]
path = root / 'notebooks/Proyecto_5_Sensor_Virtual_Penicilina.ipynb'
nb = nbformat.read(path, as_version=4)
new_cells = []
for cell in nb.cells:
    if cell.cell_type != 'code':
        new_cells.append(cell)
        continue
    narratives = [o.data['text/markdown'] for o in cell.outputs
                  if o.output_type in ('display_data', 'execute_result') and 'text/markdown' in o.get('data', {})]
    if narratives:
        source = cell.source
        tree = ast.parse(source)
        spans = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name) and node.value.func.id == 'display'
                    and node.value.args and isinstance(node.value.args[0], ast.Call)
                    and isinstance(node.value.args[0].func, ast.Name)
                    and node.value.args[0].func.id == 'Markdown'):
                spans.append((node.lineno - 1, node.end_lineno))
        lines = source.splitlines()
        for start, end in sorted(spans, reverse=True):
            del lines[start:end]
        cell.source = '\n'.join(lines).rstrip()
        cell.outputs = [o for o in cell.outputs if 'text/markdown' not in o.get('data', {})]
        if cell.source:
            new_cells.append(cell)
        for narrative in narratives:
            new_cells.append(nbformat.v4.new_markdown_cell(narrative.strip()))
    else:
        new_cells.append(cell)
nb.cells = new_cells

for cell in nb.cells:
    if cell.cell_type != 'markdown':
        continue
    if cell.source.startswith('### Interpretación de las trayectorias'):
        cell.source = '''### Interpretación de las trayectorias

En los lotes de entrenamiento, la concentración permanece cerca de cero durante las primeras horas y después aumenta. A partir de aproximadamente 150–200 horas aparecen diferencias importantes: algunos lotes continúan acumulando penicilina y otros muestran una reducción de concentración, aun sin fallas inyectadas.

Los lotes con control PAT presentan trayectorias más agrupadas en este conjunto. Los grupos Receta y Operador muestran una dispersión mayor al final. Estas diferencias describen las simulaciones disponibles; no demuestran un efecto causal de cambiar de estrategia.

La mediana muestra saltos en los extremos temporales porque los lotes terminan en momentos diferentes y cambia el grupo de observaciones que la sostiene. Esos saltos no representan cambios instantáneos de un reactor. Esta limitación también afecta a la referencia por curva típica y se tendrá en cuenta al interpretar su error.

Como la concentración tiene una evolución temporal marcada, un modelo puede obtener un R² alto simplemente reconociendo la fase. La comparación con una referencia que conoce el tiempo y la estrategia permite evaluar si las señales operativas aportan información adicional.'''
    if cell.source.startswith('### Interpretación de las señales del proceso'):
        cell.source = '''### Interpretación de las señales del proceso

En el lote 2, la alimentación de azúcar aumenta por escalones al inicio y después se mantiene durante tramos prolongados. El oxígeno disuelto presenta descensos y recuperaciones que no siguen una relación lineal simple con la concentración de penicilina.

La temperatura se mantiene la mayor parte del tiempo cerca de 25 °C y el pH alrededor de 6,5, aunque aparecen oscilaciones transitorias. Una variable controlada puede mostrar poca variación y seguir siendo relevante para describir las condiciones operativas.

El volumen aumenta y posteriormente muestra descensos repetidos. Estos cambios son compatibles con las extracciones de caldo registradas en el archivo; una caída del volumen no debe confundirse por sí sola con una falla. En este lote, la concentración de penicilina sigue aumentando hasta aproximarse a 30 g/L.

Para representar el historial reciente incluiré medias retrospectivas y cambios a una hora de alimentación, oxígeno, temperatura y CO2. Cada cálculo permanecerá dentro del lote y utilizará solo el presente y el pasado.'''
    if cell.source.startswith('### Cierre del análisis exploratorio'):
        cell.source = '''### Cierre del análisis exploratorio

La concentración presenta una correlación de aproximadamente **0,92 con el tiempo**, **0,68 con el volumen** y **0,65 con el CO2 del gas** en entrenamiento. El pH tiene una correlación cercana a cero, coherente con una señal que se mantiene controlada. Estos valores describen asociaciones y pueden estar influidos por la evolución común del lote.

La principal dificultad será distinguir la trayectoria típica de fermentación de la información adicional que aportan los sensores. Por eso el tiempo también estará disponible para la referencia y las métricas se revisarán por lote y por tramo temporal.

Conservaré los cambios de operación y no eliminaré registros por estar en los extremos de una distribución. La selección de entradas responde a su significado y disponibilidad; las correlaciones no se interpretarán como relaciones causales ni se utilizarán para escoger variables mirando la prueba.'''
    if cell.source.startswith('El grupo con mayor aumento medio del error') and 'El tiempo domina esta prueba' not in cell.source:
        cell.source += '''

El tiempo domina esta prueba de sensibilidad: al desalinearlo, el MAE aumenta aproximadamente **11,12 g/L**. Entre las señales operativas destacan **CO2 del gas** y **aireación/O2 del gas**, con aumentos de aproximadamente **0,91 y 0,48 g/L**. El modelo combina la fase del proceso con información de operación, pero sigue dependiendo mucho de la estructura temporal de la simulación.

La temperatura aporta una variación prácticamente nula en esta prueba. Esto no significa que carezca de importancia física: dentro de los lotes normales se mantiene controlada y otras señales pueden contener información relacionada.'''
    if cell.source.startswith('### Interpretación de los errores') and 'El MAE por tramo aumenta' not in cell.source:
        cell.source += '''

El MAE por tramo aumenta de **0,08 g/L antes de 50 horas** a **1,99 g/L después de 200 horas**. Por niveles de concentración, pasa de **0,30 g/L en el nivel bajo** a **1,52 g/L en el alto**. La parte inicial, cercana a cero, es más sencilla y contribuye a las buenas métricas globales.

En el lote 45, el modelo reconoce una trayectoria de menor concentración que la curva típica, pero sobreestima parte del tramo final. En el lote 33, sigue la caída general después del máximo con oscilaciones que no aparecen en la referencia suave del simulador. Estas diferencias justifican mantener los gráficos junto a las métricas y no presentar una precisión uniforme.'''
    if cell.source.startswith('# 10. Conclusiones finales'):
        cell.source = '# 10. Conclusiones finales'
    if cell.source.startswith('El proyecto permitió construir y evaluar') and 'El resultado depende especialmente' not in cell.source:
        cell.source += '''

El resultado depende especialmente del tiempo transcurrido y pierde precisión al final del lote. En consecuencia, la utilidad demostrada es estimar la concentración en **lotes normales de estrategias conocidas dentro de IndPenSim**. La evaluación con datos de planta y la comparación con mediciones de laboratorio independientes quedan fuera de este proyecto.'''

# Se eliminan solo metadatos de duración de ejecución, no resultados ni conteos.
for cell in nb.cells:
    cell.metadata.pop('execution', None)
nbformat.validate(nb)
nbformat.write(nb, path)
exporter = HTMLExporter(template_name='lab')
body, _ = exporter.from_notebook_node(nb)
(root / 'Proyecto_5_Sensor_Virtual_Penicilina.html').write_text(body, encoding='utf-8')
print(f'Notebook final: {len(nb.cells)} celdas. Vista HTML creada.')
