"""Construye los notebooks reproducibles; el análisis se ejecuta dentro de cada notebook."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


SETUP = '''
from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

%matplotlib inline
SEED = 12345
ruta_proyecto = Path.cwd()
if ruta_proyecto.name == 'notebooks':
    ruta_proyecto = ruta_proyecto.parent
assert (ruta_proyecto / 'data/source/provenance.json').is_file()
ruta_export = ruta_proyecto / 'exports_powerbi'
for carpeta in ['images', 'exports_powerbi', 'verification', 'models', 'data/processed']:
    (ruta_proyecto / carpeta).mkdir(parents=True, exist_ok=True)
pd.set_option('display.max_columns', 12)
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
plt.rcParams.update({'figure.figsize': (12, 5), 'figure.dpi': 110,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.18, 'font.size': 10,
                     'axes.titleweight': 'bold', 'axes.titlepad': 12,
                     'axes.prop_cycle': plt.cycler(color=['#16697A', '#D97736', '#6A4C93', '#557A46'])})

def guardar_figura(nombre):
    plt.tight_layout()
    plt.savefig(ruta_proyecto / 'images' / nombre, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

def numero(valor, decimales=2):
    return f'{valor:,.{decimales}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')

print('Entorno preparado. Todas las rutas del proyecto son relativas.')
'''

LOAD = '''
# Este extracto contiene las primeras 37 posiciones del CSV original, sin alterar valores.
original = pd.read_csv(ruta_proyecto / 'data/source/indpensim_process_original.csv.gz')
estadisticas_original = pd.read_csv(ruta_proyecto / 'data/source/100_Batches_IndPenSim_Statistics.csv')
procedencia = json.loads((ruta_proyecto / 'data/source/provenance.json').read_text(encoding='utf-8'))

# La correspondencia se documenta por posición porque algunos encabezados son incorrectos.
nombres = ['tiempo_h', 'aireacion', 'agitacion_rpm', 'azucar_l_h', 'acido_l_h',
           'base_l_h', 'refrigeracion_l_h', 'calefaccion_l_h', 'agua_l_h', 'presion_bar',
           'extraccion_l_h', 'sustrato_g_l', 'oxigeno_disuelto_mg_l', 'penicilina_g_l',
           'volumen_l', 'peso_kg', 'ph', 'temperatura_k', 'calor_generado', 'co2_gas',
           'paa_alimentacion_l_h', 'paa_laboratorio', 'aceite_l_h', 'nh3_laboratorio',
           'consumo_oxigeno', 'o2_gas', 'penicilina_laboratorio_g_l', 'biomasa_laboratorio_g_l',
           'evolucion_carbono', 'amonio_kg', 'viscosidad_laboratorio', 'falla_activa',
           'control_operador', 'modo_pat', 'lote_repetido', 'lote', 'lote_con_falla']
assert original.shape == (113935, 37)
df = original.copy()
df.columns = nombres
for columna in ['lote', 'lote_repetido', 'falla_activa', 'lote_con_falla', 'control_operador', 'modo_pat']:
    assert df[columna].eq(df[columna].round()).all()
    df[columna] = df[columna].astype(int)
df = df.sort_values(['lote', 'tiempo_h']).reset_index(drop=True)
assert df.lote.eq(df.lote_repetido).all()
df['estrategia'] = np.select([df.modo_pat.eq(2), df.control_operador.eq(1)],
                             ['PAT', 'Operador'], default='Receta')
df['temperatura_c'] = df.temperatura_k - 273.15
print('Datos de proceso:', original.shape)
print('Tabla de estadísticas:', estadisticas_original.shape)
display(df[['lote', 'tiempo_h', 'penicilina_g_l', 'temperatura_c', 'ph', 'estrategia']].head())
'''

SPLIT = '''
# Excluyo los lotes 91-100 con fallas inyectadas del experimento principal.
lotes = df.groupby('lote').agg(estrategia=('estrategia', 'first'),
                             lote_con_falla=('lote_con_falla', 'first'),
                             registros=('tiempo_h', 'size'), duracion_h=('tiempo_h', 'max'))
normales = lotes.index[lotes.lote_con_falla.eq(0)].to_numpy()
train_ids, resto_ids = train_test_split(normales, test_size=0.4, random_state=SEED,
                                       stratify=lotes.loc[normales, 'estrategia'])
val_ids, test_ids = train_test_split(resto_ids, test_size=0.5, random_state=SEED,
                                    stratify=lotes.loc[resto_ids, 'estrategia'])
fault_ids = lotes.index[lotes.lote_con_falla.eq(1)].to_numpy()
particiones = {'Entrenamiento': train_ids, 'Validación': val_ids,
               'Prueba normal': test_ids, 'Excluidos por fallas': fault_ids}
lotes['particion'] = ''
for nombre, ids in particiones.items():
    lotes.loc[ids, 'particion'] = nombre
assert set(train_ids).isdisjoint(val_ids) and set(train_ids).isdisjoint(test_ids)
assert set(val_ids).isdisjoint(test_ids)
assert len(set(train_ids) | set(val_ids) | set(test_ids) | set(fault_ids)) == 100
df['particion'] = df.lote.map(lotes.particion)
display(lotes.groupby(['particion', 'estrategia']).agg(lotes=('registros', 'size'),
                                                     registros=('registros', 'sum')))
'''

FEATURES = '''
# Utilizo señales operativas; excluyo las referencias internas y las mediciones offline.
sensores = ['aireacion', 'azucar_l_h', 'acido_l_h', 'base_l_h', 'refrigeracion_l_h',
            'calefaccion_l_h', 'agua_l_h', 'presion_bar', 'extraccion_l_h',
            'oxigeno_disuelto_mg_l', 'volumen_l', 'peso_kg', 'ph', 'temperatura_c',
            'co2_gas', 'paa_alimentacion_l_h', 'aceite_l_h', 'o2_gas']
X = df[['tiempo_h'] + sensores].copy()
# El modo de control se conoce durante la operación; no lo deduzco del identificador del lote.
X['control_operador'] = df.control_operador
X['control_pat'] = df.modo_pat.eq(2).astype(int)
historicas = ['azucar_l_h', 'oxigeno_disuelto_mg_l', 'temperatura_c', 'co2_gas']
for columna in historicas:
    grupo = df.groupby('lote')[columna]
    # Diez lecturas: ventana retrospectiva de aproximadamente dos horas, incluido el presente.
    X[columna + '_media_2h'] = grupo.transform(lambda s: s.rolling(10, min_periods=1).mean())
    X[columna + '_cambio_1h'] = grupo.diff(5)
y = df.penicilina_g_l.copy()
prohibidas = {'lote', 'lote_repetido', 'lote_con_falla', 'falla_activa', 'penicilina_g_l',
              'penicilina_laboratorio_g_l', 'sustrato_g_l', 'particion'}
assert set(X).isdisjoint(prohibidas)
assert y.notna().all() and not np.isinf(X.to_numpy()).any()
mascaras = {nombre: df.lote.isin(ids) for nombre, ids in particiones.items()}
tr, va, te, fa = [mascaras[n] for n in particiones]
print('Características:', X.shape[1])
print('Ausencias generadas al inicio de cada lote:', int(X.isna().sum().sum()))
'''

UTILS = '''
def predecir(modelo, datos):
    # La concentración no puede ser negativa. Aplico la misma regla a todos los modelos.
    return np.maximum(0, modelo.predict(datos))

def evaluar(real, predicho, ids):
    tabla = pd.DataFrame({'real': np.asarray(real), 'predicho': np.asarray(predicho),
                          'lote': np.asarray(ids)})
    tabla['error_absoluto'] = (tabla.real - tabla.predicho).abs()
    return {'MAE_lotes': tabla.groupby('lote').error_absoluto.mean().mean(),
            'MAE': mean_absolute_error(real, predicho),
            'RMSE': np.sqrt(mean_squared_error(real, predicho)),
            'R2': r2_score(real, predicho)}

def ajustar_referencia(datos):
    # Curvas típicas de concentración, aprendidas solo con los lotes de entrenamiento.
    return {e: g.groupby('tiempo_h').penicilina_g_l.median().sort_index()
            for e, g in datos.groupby('estrategia')}

def predecir_referencia(curvas, datos):
    resultado = pd.Series(index=datos.index, dtype=float)
    for estrategia, grupo in datos.groupby('estrategia'):
        curva = curvas[estrategia]
        resultado.loc[grupo.index] = np.interp(grupo.tiempo_h, curva.index, curva.values)
    return resultado.to_numpy()
'''

main = [md('''
# Proyecto 5 - Estimación de la concentración de penicilina en un proceso biofarmacéutico simulado

La **fermentación** permite producir sustancias de interés industrial mediante la actividad de microorganismos. En la fabricación de penicilina, las condiciones de alimentación, aireación, temperatura y pH influyen en la evolución del cultivo y en la formación del producto.

Desde el punto de vista operativo, conocer la concentración de penicilina durante un lote permite seguir su evolución. Sin embargo, una medición de laboratorio puede estar disponible con menor frecuencia que las señales del proceso. Un **sensor virtual** utiliza esas señales para estimar una variable que no se mide continuamente.

**El objetivo principal de este proyecto es evaluar si puedo estimar la concentración actual de penicilina en lotes no utilizados para entrenar el modelo.** Trabajaré con IndPenSim, un conjunto de datos generado mediante simulación matemática de un fermentador industrial de 100.000 litros. Los registros no son mediciones de una planta real.

Primero revisaré la estructura y calidad de los datos. Después analizaré el proceso, construiré una referencia sencilla y compararé modelos de regresión. La finalidad es comprender qué precisión alcanzan, en qué fases se equivocan y qué limitaciones tendría su aplicación.

El proyecto se limita al sensor virtual en operación sin fallas inyectadas. Los diez lotes con fallas quedan excluidos del modelado y de la evaluación predictiva. Este notebook estima el estado actual; no presenta un pronóstico de concentración futura ni una predicción de la fecha de mantenimiento.
'''), md('''
## Fuente de los datos y alcance

- [Datos originales de Stephen Goldrick — Mendeley Data, versión 2](https://data.mendeley.com/datasets/pdnjz7zz5x/2), DOI **10.17632/pdnjz7zz5x.2**, licencia **CC BY 4.0**.
- [Notebook de referencia del autor](https://github.com/StephenGoldie/indpensim-notebook).
- Goldrick y colaboradores (2015), *The development of an industrial-scale fed-batch fermentation simulation*, Journal of Biotechnology, 193, 70–82.
- Goldrick y colaboradores (2019), *Modern day monitoring and control challenges outlined on an industrial-scale benchmark fermentation process*, Computers & Chemical Engineering, 130, 106471.

El archivo principal contiene 100 lotes con distintas duraciones y estrategias de control. Los primeros 90 corresponden a operación sin fallas inyectadas; los últimos 10 contienen desviaciones simuladas. Una falla inyectada no significa necesariamente que el lote termine con baja concentración.

Para que el proyecto sea manejable, incluyo un **extracto fiel de las primeras 37 posiciones del CSV**, sin eliminar filas ni imputar datos. El bloque Raman no forma parte de este experimento. El archivo de procedencia registra el nombre, tamaño y SHA-256 del original; el script `scripts/extract_process.py` permite repetir la extracción.

Conservaré las estadísticas de producción total como fuente auxiliar. No utilizaré sus columnas rotuladas en kilogramos como objetivos, porque su interpretación y conciliación con concentración y volumen requieren aclaración.
'''), md('''
# 1. Cargo los archivos y reviso su estructura

## 1.1 Importo las librerías y preparo las rutas

Utilizaré rutas relativas para poder ejecutar el notebook desde la carpeta del proyecto o desde `notebooks`. Mantendré una semilla fija y limitaré el número de configuraciones para que la comparación sea reproducible.
'''), code(SETUP), md('''
## 1.2 Cargo los datos y documento la correspondencia de columnas

El CSV original tiene un problema de encabezados: algunos nombres no describen los valores situados debajo y sobran dos nombres al final. Por esa razón reviso la correspondencia **por posición**, antes de seleccionar identificadores o etiquetas.

La posición 35, contando desde cero, identifica los lotes 1–100; la posición 34 repite esos identificadores. La posición 36 distingue lotes con fallas y coincide con la tabla de estadísticas. La posición 31 contiene la activación temporal de la falla. Los nombres originales `Batch ID` y `Fault flag`, situados después, no se utilizan: contienen señales numéricas incompatibles con esos significados.

Esta corrección se aplica a una copia del extracto. No modifica los archivos originales.
'''), code(LOAD), code('''
descripciones = [
    'Tiempo transcurrido desde el inicio del lote, h', 'Aireación, escala original (el encabezado indica L/h)',
    'Velocidad del agitador, rpm', 'Alimentación de azúcar, L/h', 'Adición de ácido, L/h',
    'Adición de base, L/h', 'Caudal de refrigeración, L/h', 'Caudal de calefacción, L/h',
    'Agua de dilución, L/h', 'Presión de cabeza, bar', 'Extracción de caldo, signo original negativo, L/h',
    'Concentración de sustrato del simulador, g/L', 'Oxígeno disuelto, mg/L',
    'Concentración de penicilina del simulador, g/L', 'Volumen del caldo, L', 'Peso del recipiente/caldo según fuente, kg',
    'pH', 'Temperatura, K', 'Calor generado, escala original', 'CO2 del gas de salida, escala original',
    'Alimentación de ácido fenilacético (PAA), L/h', 'PAA de laboratorio, unidades pendientes de aclarar',
    'Alimentación de aceite, L/h', 'NH3 de laboratorio, unidades pendientes de aclarar',
    'Consumo de oxígeno, escala original', 'O2 del gas de salida, escala original',
    'Penicilina de laboratorio, g/L', 'Biomasa de laboratorio, g/L',
    'Evolución de carbono, escala original', 'Adiciones de amonio, kg', 'Viscosidad de laboratorio, centipoise según fuente',
    'Intervalo con falla activa, 0/1', 'Control por operador, 0/1', 'Modo de operación: 1 convencional, 2 PAT',
    'Identificador de lote redundante', 'Identificador validado del lote', 'Lote que contiene alguna falla, 0/1']
diccionario = pd.DataFrame({'posicion_csv_desde_cero': range(37),
                            'nombre_original': original.columns,
                            'variable': nombres, 'descripcion': descripciones})
display(diccionario[['posicion_csv_desde_cero', 'variable', 'descripcion']])
'''), md('''
### Cómo interpreto las variables

La **alimentación de azúcar y aceite** aporta información sobre las entradas al cultivo. La **aireación y el oxígeno disuelto** describen parte de su entorno de operación. La **temperatura, el pH y los caudales de regulación** permiten observar cómo se mantiene el proceso.

PAA corresponde a ácido fenilacético y PAT a tecnología analítica de procesos. Las señales de gas se conservarán en su escala publicada: un encabezado con porcentaje no basta para decidir si los valores se almacenan como fracción o porcentaje. Tampoco convertiré la aireación en un balance de masa sin verificar su unidad.

La concentración continua de penicilina es la **referencia del simulador**. La utilizaré para entrenar y evaluar, pero no como entrada. El sustrato, el calor generado y los análisis offline tampoco entrarán en el modelo principal, para evitar depender de estados internos o mediciones cuya disponibilidad inmediata no está confirmada.
'''), md('''
## 1.3 Compruebo ausentes, continuidad temporal y etiquetas

Reviso cada lote por separado. El reinicio del tiempo al comenzar el siguiente lote es esperado; no voy a concatenar los lotes como si fueran una única serie continua.
'''), code('''
calidad = pd.DataFrame({'variable': df.columns, 'ausentes': df.isna().sum().values,
                        'porcentaje_ausente': df.isna().mean().values * 100,
                        'valores_distintos': df.nunique().values})
pasos = df.groupby('lote').tiempo_h.diff()
assert np.isclose(pasos.dropna(), 0.2).all()
assert not df.duplicated(['lote', 'tiempo_h']).any()
assert not np.isinf(original.to_numpy()).any()
assert df.groupby('lote').tiempo_h.min().eq(0.2).all()
assert df.groupby('lote').lote_con_falla.nunique().eq(1).all()
etiquetas_resumen = estadisticas_original.set_index('Batch ref').iloc[:, -1]
assert df.groupby('lote').lote_con_falla.first().eq(etiquetas_resumen).all()
assert df.loc[df.lote_con_falla.eq(0), 'falla_activa'].eq(0).all()
display(calidad.loc[(calidad.ausentes > 0) | (calidad.valores_distintos == 1)])
display(df.groupby('lote').agg(registros=('tiempo_h', 'size'),
                              duracion_h=('tiempo_h', 'max')).describe())
print('Duplicados en las primeras 37 columnas:', int(original.duplicated().sum()))
print('Consumo de oxígeno con valores negativos:', int(df.consumo_oxigeno.lt(0).sum()))
'''), md('''
### Hallazgos de la revisión inicial

Los **113.935 registros** corresponden a **100 lotes**, con entre **835 y 1.450 observaciones** y duraciones de **167 a 290 horas**. La frecuencia interna es constante: una lectura cada **0,2 horas, equivalentes a 12 minutos**. No hay tiempos duplicados ni huecos internos.

Cinco variables offline tienen **2.062 mediciones cada una**, por lo que el **98,19 %** de sus registros está vacío. No interpretaré esos vacíos como ceros ni completaré resultados de laboratorio con información posterior. Las variables de operación seleccionadas y el objetivo están completos.

La agitación permanece en 100 rpm y las adiciones de amonio son cero; ambas carecen de variación para este experimento. Hay **1.038 valores negativos de consumo de oxígeno**. Conservo la señal original, pero la excluyo del modelo principal, junto con otras magnitudes derivadas de interpretación incierta. El caudal de extracción negativo se conserva porque representa la convención de salida del archivo; no lo trataré automáticamente como error.
'''), md('''
# 2. Defino los lotes que utilizaré en cada etapa

Antes del análisis gráfico detallado reservo las muestras de prueba. Utilizo **54 lotes normales para entrenamiento, 18 para validación y 18 para prueba**, con representación de las tres estrategias de control en cada partición. Los **10 lotes con fallas** quedan excluidos del experimento. Sus etiquetas sirven únicamente para delimitar el alcance de operación normal.

Cada lote completo permanece en una sola partición. La división es aleatoria y estratificada **entre lotes**, no entre filas; evalúa nuevos lotes de estrategias conocidas. Como no hay fechas de campañas, no la presentaré como una validación cronológica de una planta real.

Las gráficas exploratorias de las siguientes secciones utilizarán únicamente entrenamiento. La inspección inicial del archivo permitió definir el alcance, pero la elección de modelos no consultará los objetivos de prueba.
'''), code(SPLIT), md('''
# 3. Análisis exploratorio del proceso

## 3.1 Observo la evolución de la concentración

Comparo las trayectorias de entrenamiento y su mediana por estrategia. Utilizo horas transcurridas, sin normalizar por la duración final del lote: esa duración no se conoce necesariamente al emitir una estimación.
'''), code('''
eda = df[df.lote.isin(train_ids)].copy()
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
for ax, estrategia in zip(axes, ['Receta', 'Operador', 'PAT']):
    datos = eda[eda.estrategia.eq(estrategia)]
    for _, lote_datos in datos.groupby('lote'):
        ax.plot(lote_datos.tiempo_h, lote_datos.penicilina_g_l, color='#16697A', alpha=0.16, lw=0.8)
    curva = datos.groupby('tiempo_h').penicilina_g_l.median()
    ax.plot(curva.index, curva, color='#D97736', lw=2.4, label='Mediana de entrenamiento')
    ax.set(title=estrategia, xlabel='Tiempo del lote (h)')
    ax.legend(fontsize=8)
axes[0].set_ylabel('Penicilina (g/L)')
fig.suptitle('Evolución de la concentración por estrategia de control', y=1.03)
guardar_figura('01_concentracion_entrenamiento.png')
finales_eda = eda.groupby('lote').tail(1)
display(finales_eda.groupby('estrategia').penicilina_g_l.agg(['count', 'median', 'min', 'max']))
'''), md('''
### Interpretación de las trayectorias

La concentración cambia con el tiempo de fermentación, pero los lotes no siguen una única trayectoria. Por eso un modelo podría parecer preciso al reconocer solamente la fase del proceso. Para comprobar el valor adicional de los sensores, compararé los modelos con una curva típica aprendida de entrenamiento.

La separación por estrategia describe diferencias entre estas simulaciones. No demuestra que cambiar de estrategia cause una mejora determinada: también cambian las condiciones y la evolución de los lotes. En los extremos temporales, las curvas se apoyan en menos lotes porque las duraciones son diferentes.
'''), md('''
## 3.2 Reviso las señales operativas durante un lote

Selecciono el primer lote de entrenamiento por identificador para ilustrar el proceso, sin escogerlo por el resultado del modelo. Represento alimentación, oxígeno, temperatura y pH en sus unidades correspondientes.
'''), code('''
lote_ejemplo = int(min(train_ids))
ejemplo = eda[eda.lote.eq(lote_ejemplo)]
fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
variables_grafico = [('azucar_l_h', 'Alimentación de azúcar (L/h)'),
                    ('oxigeno_disuelto_mg_l', 'Oxígeno disuelto (mg/L)'),
                    ('temperatura_c', 'Temperatura (°C)'), ('ph', 'pH'),
                    ('volumen_l', 'Volumen (L)'), ('penicilina_g_l', 'Penicilina (g/L)')]
for ax, (variable, titulo) in zip(axes.flat, variables_grafico):
    ax.plot(ejemplo.tiempo_h, ejemplo[variable], lw=1.2)
    ax.set(title=titulo, xlabel='Tiempo (h)')
fig.suptitle(f'Condiciones de operación del lote {lote_ejemplo}', y=1.01)
guardar_figura('02_senales_lote_entrenamiento.png')
'''), md('''
### Interpretación de las señales del proceso

La alimentación y los caudales de regulación pueden cambiar durante el lote, mientras que temperatura y pH se mantienen cerca de sus condiciones de control. Una variable controlada puede mostrar poca variación y seguir siendo relevante para describir la operación.

La relación entre estas señales y la concentración no tiene por qué ser instantánea. Incorporaré resúmenes del historial reciente de alimentación, oxígeno, temperatura y CO2 para representar parte de esa evolución. Cada resumen se calculará dentro del lote y únicamente hacia atrás en el tiempo.
'''), md('''
## 3.3 Comparo asociaciones entre las variables principales

La matriz de correlación sirve para explorar asociaciones lineales. La interpreto con cuidado porque las filas de un lote están relacionadas temporalmente y varias señales cambian al mismo tiempo que avanza la fermentación.
'''), code('''
cols_corr = ['tiempo_h', 'azucar_l_h', 'oxigeno_disuelto_mg_l', 'volumen_l',
             'temperatura_c', 'ph', 'co2_gas', 'penicilina_g_l']
etiquetas_corr = ['Tiempo', 'Azúcar', 'O2 disuelto', 'Volumen', 'Temperatura', 'pH', 'CO2 gas', 'Penicilina']
corr = eda[cols_corr].corr()
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(cols_corr)), etiquetas_corr, rotation=40, ha='right')
ax.set_yticks(range(len(cols_corr)), etiquetas_corr)
for i in range(len(cols_corr)):
    for j in range(len(cols_corr)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                color='white' if abs(corr.iloc[i, j]) > 0.6 else '#243447', fontsize=9)
fig.colorbar(im, ax=ax, label='Correlación de Pearson')
ax.set_title('Asociaciones en los lotes de entrenamiento')
guardar_figura('03_correlaciones.png')
'''), md('''
### Cierre del análisis exploratorio

Los datos permiten plantear una regresión supervisada con una referencia continua de concentración. La principal dificultad es distinguir la evolución típica del lote de la información adicional que aportan las condiciones operativas.

No eliminaré observaciones simplemente por estar en los extremos de una distribución. Conservaré los cambios de operación y excluiré únicamente variables por razones documentadas de disponibilidad, significado o falta de variación. La correlación no se utilizará para afirmar causalidad ni para escoger variables mirando los lotes de prueba.
'''), md('''
# 4. Preparo los datos para el sensor virtual

Utilizaré 18 señales operativas, el tiempo transcurrido, dos indicadores del modo de control y ocho características históricas. Los cambios a una hora quedan ausentes durante las primeras cinco lecturas de cada lote; la mediana para completarlos se aprenderá exclusivamente dentro de entrenamiento.

No incluiré la concentración actual ni retardos de penicilina, el identificador del lote, las etiquetas de falla, la duración final ni los resultados de producción. Tampoco utilizaré análisis de laboratorio asociados a un momento sin confirmar cuándo estarían disponibles.

Las señales seleccionadas se consideran disponibles al final de cada intervalo de 12 minutos en este experimento. Esa es una hipótesis operativa que tendría que comprobarse con instrumentación real. La curva objetivo procede del simulador y no valida por sí sola un sensor industrial.
'''), code(FEATURES), code('''
# Verifico la causalidad de las variables históricas en un prefijo de lote.
prefijo = df[df.lote.eq(lote_ejemplo)].head(25)
for columna in historicas:
    esperado = prefijo[columna].rolling(10, min_periods=1).mean()
    np.testing.assert_allclose(X.loc[prefijo.index, columna + '_media_2h'], esperado)
    assert X.loc[df.groupby('lote').head(5).index, columna + '_cambio_1h'].isna().all()
diccionario['uso'] = np.where(diccionario.variable.isin(sensores + ['tiempo_h', 'control_operador', 'modo_pat']),
                              'Entrada operativa o configuración', 'Conservada, no predictora')
diccionario.loc[diccionario.variable.eq('temperatura_k'), 'uso'] = 'Entrada tras convertir K a °C'
diccionario.loc[diccionario.variable.eq('penicilina_g_l'), 'uso'] = 'Objetivo del simulador'
diccionario.loc[diccionario.variable.isin(['lote', 'lote_repetido']), 'uso'] = 'Identificación, nunca predictor'
diccionario.loc[diccionario.variable.isin(['falla_activa', 'lote_con_falla']), 'uso'] = 'Exclusión de lotes con fallas, nunca predictor'
display(pd.DataFrame({'variable_modelo': X.columns, 'ausentes': X.isna().sum().values}))
'''), md('''
# 5. Defino las métricas y construyo la referencia

La métrica principal será el **MAE medio entre lotes**: calculo el error absoluto medio dentro de cada lote y después doy el mismo peso a todos los lotes. De esta manera, un lote largo no domina la selección.

También mostraré MAE y RMSE sobre todas las filas y R². El error se expresa en g/L; evitaré MAPE porque la concentración se aproxima a cero al inicio. Un R² alto sobre todas las filas puede reflejar en parte la diferencia entre fases, por lo que revisaré errores por lote y por tramo temporal.

La referencia utiliza la mediana de concentración a cada tiempo y estrategia en entrenamiento. Fuera de su rango temporal mantiene el valor extremo disponible. Es un modelo sencillo y explícito que no necesita sensores ni resultados anteriores del lote evaluado.
'''), code(UTILS), code('''
referencia = ajustar_referencia(df[tr])
pred_ref_val = predecir_referencia(referencia, df[va])
metricas_referencia = evaluar(y[va], pred_ref_val, df.loc[va, 'lote'])
display(pd.DataFrame([{'Modelo': 'Curva típica', **metricas_referencia}]))
'''), md('''
# 6. Entreno y comparo modelos

Compararé una regresión lineal regularizada (**Ridge**), **Random Forest**, **HistGradientBoosting** y **XGBoost**. Ridge ayuda a establecer una referencia lineal; los modelos de árboles permiten representar relaciones no lineales.

Probaré una configuración de Ridge y Random Forest y dos configuraciones moderadas para cada método de boosting. Desactivo la parada temprana automática basada en una separación aleatoria de filas. La selección se realiza sobre los 18 lotes completos de validación, utilizando el MAE medio entre lotes.

No utilizaré el conjunto de prueba para elegir parámetros. Las predicciones negativas se limitarán a cero de manera uniforme, por el significado físico de la concentración.
'''), code('''
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor

candidatos = {
    'Ridge': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=10)),
    'Random Forest': make_pipeline(SimpleImputer(strategy='median'),
        RandomForestRegressor(n_estimators=120, max_depth=16, min_samples_leaf=8,
                              max_features=0.9, n_jobs=4, random_state=SEED)),
    'HistGB 15 hojas': make_pipeline(SimpleImputer(strategy='median'),
        HistGradientBoostingRegressor(max_iter=200, learning_rate=0.07, max_leaf_nodes=15,
                                      l2_regularization=5, early_stopping=False, random_state=SEED)),
    'HistGB 31 hojas': make_pipeline(SimpleImputer(strategy='median'),
        HistGradientBoostingRegressor(max_iter=200, learning_rate=0.07, max_leaf_nodes=31,
                                      l2_regularization=5, early_stopping=False, random_state=SEED)),
    'XGBoost profundidad 4': make_pipeline(SimpleImputer(strategy='median'),
        XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.06, subsample=0.85,
                     colsample_bytree=0.9, reg_lambda=5, objective='reg:squarederror',
                     tree_method='hist', n_jobs=4, random_state=SEED)),
    'XGBoost profundidad 6': make_pipeline(SimpleImputer(strategy='median'),
        XGBRegressor(n_estimators=250, max_depth=6, learning_rate=0.06, subsample=0.85,
                     colsample_bytree=0.9, reg_lambda=5, objective='reg:squarederror',
                     tree_method='hist', n_jobs=4, random_state=SEED))}
resultados_val = [{'Modelo': 'Curva típica', **metricas_referencia, 'segundos': 0}]
ajustados = {}
for nombre, modelo in candidatos.items():
    inicio = time.perf_counter()
    modelo.fit(X[tr], y[tr])
    pred = predecir(modelo, X[va])
    resultados_val.append({'Modelo': nombre, **evaluar(y[va], pred, df.loc[va, 'lote']),
                           'segundos': time.perf_counter() - inicio})
    ajustados[nombre] = modelo
    print(nombre, 'completado.')
validacion = pd.DataFrame(resultados_val).sort_values('MAE_lotes').reset_index(drop=True)
ganador = validacion.iloc[0].Modelo
display(validacion)
print('Seleccionado con validación:', ganador)
'''), code('''
fig, ax = plt.subplots(figsize=(11, 5))
ax.barh(validacion.Modelo[::-1], validacion.MAE_lotes[::-1], color='#16697A')
ax.set(xlabel='MAE medio entre lotes (g/L)', title='Comparación de modelos en validación')
guardar_figura('04_comparacion_validacion.png')
mejora_val = 100 * (1 - validacion.iloc[0].MAE_lotes / metricas_referencia['MAE_lotes'])
display(Markdown(f"""### Resultado de la selección

El menor MAE medio entre lotes corresponde a **{ganador}**, con **{numero(validacion.iloc[0].MAE_lotes)} g/L**. La referencia por tiempo y estrategia obtiene **{numero(metricas_referencia['MAE_lotes'])} g/L**; la diferencia relativa es **{numero(mejora_val, 1)} %**.

Esta selección se basa únicamente en validación. El siguiente paso comprueba su estabilidad dentro de entrenamiento antes de abrir la prueba final."""))
'''), md('''
## 6.1 Compruebo estabilidad con tres particiones de lotes

Mantengo fija la configuración seleccionada y repito el ajuste dentro de los 54 lotes de entrenamiento, separándolos en tres grupos estratificados por estrategia. Ningún lote de validación externa ni de prueba participa en estos ajustes.

Esta comprobación describe la sensibilidad a los lotes de entrenamiento; no es una validación anidada ni una nueva búsqueda de modelos. Cada fila de resultados compara el modelo y la referencia sobre los mismos lotes.
'''), code('''
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
cv_resultados = []
ids_ordenados = np.sort(train_ids)
for corte, (i_train, i_val) in enumerate(cv.split(ids_ordenados, lotes.loc[ids_ordenados, 'estrategia']), 1):
    m_train = df.lote.isin(ids_ordenados[i_train])
    m_val = df.lote.isin(ids_ordenados[i_val])
    curva_cv = ajustar_referencia(df[m_train])
    ref_cv = predecir_referencia(curva_cv, df[m_val])
    if ganador == 'Curva típica':
        pred_cv = ref_cv
    else:
        modelo_cv = clone(candidatos[ganador]).fit(X[m_train], y[m_train])
        pred_cv = predecir(modelo_cv, X[m_val])
    met = evaluar(y[m_val], pred_cv, df.loc[m_val, 'lote'])
    met_ref = evaluar(y[m_val], ref_cv, df.loc[m_val, 'lote'])
    cv_resultados.append({'corte': corte, **met, 'MAE_lotes_referencia': met_ref['MAE_lotes'],
                          'mejora_porcentaje': 100 * (1 - met['MAE_lotes'] / met_ref['MAE_lotes'])})
cv_tabla = pd.DataFrame(cv_resultados)
display(cv_tabla)
display(Markdown(f"""### Interpretación de la estabilidad

El MAE medio entre lotes varía entre **{numero(cv_tabla.MAE_lotes.min())} y {numero(cv_tabla.MAE_lotes.max())} g/L** en los tres cortes. El modelo supera la referencia en **{int(cv_tabla.mejora_porcentaje.gt(0).sum())} de 3** particiones.

Las diferencias entre cortes muestran cuánto influye la composición de los lotes disponibles. No sumaré estas filas como si fueran nuevas campañas independientes ni utilizaré estos resultados para modificar la prueba reservada."""))
'''), md('''
## 6.2 Analizo qué grupos de variables utiliza el modelo

Para evitar dar importancia artificial a una variable frente a sus retardos, desplazo conjuntamente cada señal y sus características históricas dentro de cada lote de validación. Uso tres desplazamientos de 20 %, 40 % y 60 % de su longitud y mido cuánto aumenta el error.

Esta es una **prueba de sensibilidad por desplazamiento temporal**, no una prueba causal. La alteración puede crear combinaciones poco habituales y las señales correlacionadas pueden compartir información. No utilizaré esta tabla para volver a seleccionar variables.
'''), code('''
grupos = {'Tiempo': ['tiempo_h'], 'Alimentación azúcar': ['azucar_l_h'],
          'Oxígeno disuelto': ['oxigeno_disuelto_mg_l'], 'Temperatura': ['temperatura_c'],
          'CO2 gas': ['co2_gas'], 'Volumen y peso': ['volumen_l', 'peso_kg'],
          'Regulación pH': ['ph', 'acido_l_h', 'base_l_h'],
          'Regulación térmica': ['refrigeracion_l_h', 'calefaccion_l_h'],
          'Aireación y O2 gas': ['aireacion', 'o2_gas'],
          'Otras alimentaciones': ['agua_l_h', 'paa_alimentacion_l_h', 'aceite_l_h', 'extraccion_l_h']}
for grupo, columnas in grupos.items():
    grupos[grupo] = [c for c in X if any(c == v or c.startswith(v + '_') for v in columnas)]
sensibilidad = []
if ganador != 'Curva típica':
    xv = X[va].copy()
    base_error = evaluar(y[va], predecir(ajustados[ganador], xv), df.loc[va, 'lote'])['MAE_lotes']
    for grupo, columnas in grupos.items():
        cambios = []
        for fraccion in [0.2, 0.4, 0.6]:
            alterado = xv.copy()
            for lote_id in val_ids:
                idx = df.index[df.lote.eq(lote_id)]
                desplazamiento = max(1, int(len(idx) * fraccion))
                alterado.loc[idx, columnas] = np.roll(xv.loc[idx, columnas].to_numpy(), desplazamiento, axis=0)
            error = evaluar(y[va], predecir(ajustados[ganador], alterado), df.loc[va, 'lote'])['MAE_lotes']
            cambios.append(error - base_error)
        sensibilidad.append({'grupo': grupo, 'aumento_MAE_g_l': np.mean(cambios),
                              'desviacion': np.std(cambios)})
sensibilidad = pd.DataFrame(sensibilidad, columns=['grupo', 'aumento_MAE_g_l', 'desviacion'])
if not sensibilidad.empty:
    sensibilidad = sensibilidad.sort_values('aumento_MAE_g_l', ascending=False)
    display(sensibilidad)
    fig, ax = plt.subplots(figsize=(11, 5))
    orden = sensibilidad.iloc[::-1]
    ax.barh(orden.grupo, orden.aumento_MAE_g_l, xerr=orden.desviacion, color='#16697A')
    ax.set(xlabel='Aumento del MAE medio entre lotes (g/L)', title='Sensibilidad al desalinear señales de validación')
    guardar_figura('05_sensibilidad.png')
    display(Markdown(f'El grupo con mayor aumento medio del error es **{sensibilidad.iloc[0].grupo}**. Esto indica dependencia predictiva dentro de este modelo y de esta simulación; no demuestra una relación causal.'))
'''), md('''
# 7. Evaluación final en lotes no utilizados para seleccionar el modelo

Después de fijar la configuración, vuelvo a entrenar con los **72 lotes normales de entrenamiento y validación**. Evalúo una sola configuración final sobre los **18 lotes normales de prueba**, junto con la curva típica recalculada sobre los mismos 72 lotes.

El conjunto de prueba contiene nuevos lotes de las tres estrategias conocidas. Los lotes con fallas no participan en la selección ni en esta evaluación.
'''), code('''
desarrollo = tr | va
curva_final = ajustar_referencia(df[desarrollo])
pred_ref = predecir_referencia(curva_final, df[te])
if ganador == 'Curva típica':
    modelo_final = None
    pred_final = pred_ref.copy()
else:
    modelo_final = clone(candidatos[ganador]).fit(X[desarrollo], y[desarrollo])
    pred_final = predecir(modelo_final, X[te])
metricas_prueba = pd.DataFrame([
    {'Modelo': 'Curva típica', **evaluar(y[te], pred_ref, df.loc[te, 'lote'])},
    {'Modelo': ganador, **evaluar(y[te], pred_final, df.loc[te, 'lote'])}])
predicciones = df.loc[te, ['lote', 'tiempo_h', 'estrategia']].copy()
predicciones['real_g_l'] = y[te]
predicciones['predicho_g_l'] = pred_final
predicciones['referencia_g_l'] = pred_ref
predicciones['error_g_l'] = predicciones.real_g_l - predicciones.predicho_g_l
predicciones['error_absoluto_g_l'] = predicciones.error_g_l.abs()
errores_lote = predicciones.groupby(['lote', 'estrategia']).agg(
    MAE_g_l=('error_absoluto_g_l', 'mean'), sesgo_g_l=('error_g_l', 'mean'), registros=('tiempo_h', 'size')).reset_index()
display(metricas_prueba)
display(errores_lote.sort_values('MAE_g_l'))
'''), code('''
# Intervalo del MAE medio entre lotes mediante remuestreo de lotes, no de filas.
rng = np.random.default_rng(SEED)
errores = errores_lote.MAE_g_l.to_numpy()
bootstrap = rng.choice(errores, size=(2000, len(errores)), replace=True).mean(axis=1)
ic_mae = np.quantile(bootstrap, [0.025, 0.975])
met_final = metricas_prueba.iloc[-1]
met_ref = metricas_prueba.iloc[0]
mejora_prueba = 100 * (1 - met_final.MAE_lotes / met_ref.MAE_lotes)
display(Markdown(f"""### Resultado de la prueba final

**{ganador}** obtiene un **MAE medio entre lotes de {numero(met_final.MAE_lotes)} g/L**, frente a **{numero(met_ref.MAE_lotes)} g/L** de la curva típica. La mejora relativa es **{numero(mejora_prueba, 1)} %**.

Sobre todas las observaciones, el MAE es **{numero(met_final.MAE)} g/L**, el RMSE **{numero(met_final.RMSE)} g/L** y R² **{numero(met_final.R2, 3)}**. El intervalo de remuestreo del 95 % para el MAE medio entre lotes es **[{numero(ic_mae[0])}, {numero(ic_mae[1])}] g/L**.

Este intervalo describe la variación entre los 18 lotes evaluados bajo este modelo. No es un intervalo de predicción para cada concentración ni incluye incertidumbre por pasar del simulador a una planta real."""))
'''), md('''
## 7.1 Comparo las curvas de concentración

Muestro un lote de prueba por estrategia, seleccionado por el menor identificador, y el lote con mayor MAE. Este último se elige únicamente para diagnosticar el resultado final, sin volver a ajustar el modelo.
'''), code('''
ejemplos = [int(lotes.loc[test_ids].query('estrategia == @e').index.min()) for e in ['Receta', 'Operador', 'PAT']]
peor_lote = int(errores_lote.loc[errores_lote.MAE_g_l.idxmax(), 'lote'])
ejemplos.append(peor_lote)
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
for j, (ax, lote_id) in enumerate(zip(axes.flat, ejemplos)):
    datos = predicciones[predicciones.lote.eq(lote_id)]
    ax.plot(datos.tiempo_h, datos.real_g_l, label='Referencia del simulador', lw=1.8)
    ax.plot(datos.tiempo_h, datos.predicho_g_l, label='Sensor virtual', lw=1.5)
    ax.plot(datos.tiempo_h, datos.referencia_g_l, label='Curva típica', lw=1.1, linestyle='--')
    titulo = f'Lote {lote_id} · {datos.estrategia.iloc[0]}'
    if j == 3: titulo += ' · mayor MAE'
    ax.set(title=titulo, xlabel='Tiempo (h)', ylabel='Penicilina (g/L)')
    ax.legend(fontsize=8)
guardar_figura('06_predicciones_prueba.png')
'''), md('''
## 7.2 Analizo los errores por fase, estrategia y nivel de concentración

Divido el tiempo en tramos descriptivos fijados de antemano: 0–50 h, 50–100 h, 100–150 h, 150–200 h y más de 200 h. No representan fases biológicas demostradas. Los niveles de concentración se definen con los terciles de entrenamiento y validación, sin utilizar la distribución de prueba para fijar sus límites.
'''), code('''
predicciones['tramo_h'] = pd.cut(predicciones.tiempo_h, [0, 50, 100, 150, 200, np.inf],
                                 labels=['0–50', '50–100', '100–150', '150–200', '>200'])
cortes_nivel = y[desarrollo].quantile([1/3, 2/3]).to_numpy()
predicciones['nivel_concentracion'] = pd.cut(predicciones.real_g_l, [-np.inf, *cortes_nivel, np.inf],
                                             labels=['Baja', 'Media', 'Alta'])
errores_tramo = predicciones.groupby('tramo_h', observed=True).agg(
    MAE_g_l=('error_absoluto_g_l', 'mean'), sesgo_g_l=('error_g_l', 'mean'), registros=('lote', 'size')).reset_index()
errores_nivel = predicciones.groupby('nivel_concentracion', observed=True).agg(
    MAE_g_l=('error_absoluto_g_l', 'mean'), sesgo_g_l=('error_g_l', 'mean'), registros=('lote', 'size')).reset_index()
errores_estrategia = errores_lote.groupby('estrategia').agg(MAE_lotes=('MAE_g_l', 'mean'),
                                                         lotes=('lote', 'size')).reset_index()
display(errores_tramo)
display(errores_nivel)
display(errores_estrategia)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
muestra = predicciones.iloc[::5]
axes[0].scatter(muestra.real_g_l, muestra.predicho_g_l, s=5, alpha=0.25)
lim = max(predicciones.real_g_l.max(), predicciones.predicho_g_l.max())
axes[0].plot([0, lim], [0, lim], '--', color='#D97736')
axes[0].set(xlabel='Referencia del simulador (g/L)', ylabel='Estimación (g/L)', title='Observado y estimado')
axes[1].hist(predicciones.error_g_l, bins=55, color='#16697A', alpha=0.9)
axes[1].axvline(0, color='#D97736', linestyle='--')
axes[1].set(xlabel='Referencia − estimación (g/L)', ylabel='Registros', title='Distribución de errores')
axes[2].bar(errores_tramo.tramo_h.astype(str), errores_tramo.MAE_g_l, color='#16697A')
axes[2].set(xlabel='Tramo del lote (h)', ylabel='MAE (g/L)', title='Error por tramo temporal')
guardar_figura('07_diagnosticos_prueba.png')
peor_tramo = errores_tramo.loc[errores_tramo.MAE_g_l.idxmax()]
peor_estrategia = errores_estrategia.loc[errores_estrategia.MAE_lotes.idxmax()]
sesgo = predicciones.error_g_l.mean()
display(Markdown(f"""### Interpretación de los errores

El mayor error por tramo aparece en **{peor_tramo.tramo_h} horas**, con un MAE de **{numero(peor_tramo.MAE_g_l)} g/L**. Entre estrategias, **{peor_estrategia.estrategia}** presenta el mayor MAE medio entre lotes: **{numero(peor_estrategia.MAE_lotes)} g/L**.

El promedio de referencia menos estimación es **{numero(sesgo)} g/L**. Un valor positivo indica subestimación media; uno negativo, sobreestimación. El lote con mayor error es el **{peor_lote}**, lo que permite revisar una situación desfavorable además de los ejemplos representativos.

Estas diferencias son parte del resultado: una métrica global favorable no significa que todas las fases y condiciones tengan la misma precisión. Las métricas por tramo ponderan filas; la métrica principal conserva el mismo peso por lote."""))
'''), md('''
# 8. Alcance operativo y limitaciones

El sensor virtual estima la concentración actual a partir de señales de operación. Un resultado favorable indica capacidad para reproducir la referencia de IndPenSim en lotes reservados, dentro de estrategias conocidas. No demuestra precisión sobre concentraciones medidas en una planta real.

La referencia continua de penicilina y los modelos comparten un mismo generador físico. Algunas relaciones pueden ser más estables que en una fábrica. Antes de una aplicación real habría que verificar unidades, disponibilidad de sensores, calibración, retrasos de laboratorio y rendimiento en campañas independientes.

El experimento principal se entrena y evalúa sobre lotes sin fallas inyectadas. Esto define su dominio de operación. No se evalúa detección de fallas ni se extienden las conclusiones a operación anómala.

El identificador de lote no es una fecha. La validación entre lotes no equivale a demostrar estabilidad a lo largo de años. Tampoco se optimizan recetas ni se recomiendan cambios de fabricación: la importancia predictiva no establece causalidad.
'''), md('''
# 9. Exporto los resultados para Power BI

Exportaré tablas de operación, lotes, predicciones, métricas, calidad y diccionario. La clave temporal será **lote + tiempo transcurrido**, sin inventar fechas de calendario. La concentración se resumirá mediante promedios o valores de un instante; no se sumará como producción.

El modelo se guarda junto con las tablas necesarias para reproducir la evaluación. Los nombres y unidades de las exportaciones permiten reconstruir qué observaciones participaron en cada cálculo.
'''), code('''
import joblib
import platform
import sklearn
import xgboost

tablas = {
    'operacion_proceso.csv': df[['lote', 'tiempo_h', 'estrategia', 'particion', 'penicilina_g_l'] + sensores],
    'lotes_particiones.csv': lotes.reset_index(),
    'predicciones_prueba.csv': predicciones,
    'metricas_validacion.csv': validacion,
    'metricas_prueba.csv': metricas_prueba,
    'estabilidad_lotes.csv': cv_tabla,
    'errores_por_lote.csv': errores_lote,
    'errores_por_tramo.csv': errores_tramo,
    'errores_por_nivel.csv': errores_nivel,
    'errores_por_estrategia.csv': errores_estrategia,
    'sensibilidad_variables.csv': sensibilidad,
    'calidad_datos.csv': calidad,
    'diccionario_variables.csv': diccionario}
for nombre, tabla in tablas.items():
    tabla.to_csv(ruta_export / nombre, index=False, encoding='utf-8-sig')
df.to_csv(ruta_proyecto / 'data/processed/proceso_preparado.csv.gz', index=False, compression='gzip')
joblib.dump({'modelo': modelo_final, 'curvas': curva_final, 'caracteristicas': X.columns.tolist(),
             'nombre': ganador, 'semilla': SEED}, ruta_proyecto / 'models/sensor_virtual.joblib')
resumen = {'modelo': ganador, 'validacion': validacion.to_dict(orient='records'),
           'prueba': metricas_prueba.to_dict(orient='records'),
           'ic95_mae_lotes': ic_mae.tolist(), 'mejora_prueba_porcentaje': float(mejora_prueba),
           'cv': cv_tabla.to_dict(orient='records'), 'peor_lote': peor_lote,
           'filas': len(df), 'caracteristicas': X.shape[1],
           'particiones': {k: sorted(map(int, v)) for k, v in particiones.items()},
           'versiones': {'python': platform.python_version(), 'pandas': pd.__version__,
                         'numpy': np.__version__, 'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__}}
(ruta_proyecto / 'verification/resultados_principal.json').write_text(
    json.dumps(resumen, ensure_ascii=False, indent=2), encoding='utf-8')
print('Tablas exportadas:', len(tablas))
'''), md('''
# 10. Conclusiones finales

Las conclusiones siguientes se calculan a partir de las métricas de esta ejecución. Mantienen separados el resultado predictivo, la variación entre lotes y los límites del origen simulado.
'''), code('''
balance = ('mejora la referencia' if mejora_prueba > 0 else 'no mejora la referencia')
display(Markdown(f"""El proyecto permitió construir y evaluar un **sensor virtual de concentración de penicilina** sobre datos simulados de fabricación biofarmacéutica. La auditoría confirmó 100 lotes con frecuencia regular, una variable objetivo completa y problemas de encabezados que fue necesario documentar antes de modelar.

En los **18 lotes normales reservados**, **{ganador} {balance}**. Su MAE medio entre lotes es **{numero(met_final.MAE_lotes)} g/L**, frente a **{numero(met_ref.MAE_lotes)} g/L** de la curva típica, con una diferencia relativa de **{numero(mejora_prueba, 1)} %**. El RMSE global es **{numero(met_final.RMSE)} g/L** y R² **{numero(met_final.R2, 3)}**.

La comprobación interna supera la referencia en **{int(cv_tabla.mejora_porcentaje.gt(0).sum())} de tres cortes**. El intervalo de remuestreo del MAE de prueba, **[{numero(ic_mae[0])}, {numero(ic_mae[1])}] g/L**, refleja variación entre lotes y evita tratar las miles de lecturas como experimentos independientes.

El error no es uniforme: el tramo **{peor_tramo.tramo_h} horas** y la estrategia **{peor_estrategia.estrategia}** requieren especial atención dentro de esta prueba. Por eso la interpretación incluye curvas, sesgo, errores por fase y el lote con mayor desviación, además del resultado global.

La aplicación propuesta es **apoyo al seguimiento de la fermentación**, bajo el supuesto de disponibilidad de las señales operativas. No se ha demostrado sustitución de análisis de laboratorio, validez en una planta real ni capacidad para anticipar concentraciones futuras. Los lotes con fallas quedan excluidos: no se ha evaluado detección de anomalías ni rendimiento del sensor bajo fallas."""))
'''), md('''
# 11. Verificación final

Vuelvo a leer las exportaciones y compruebo claves, cobertura, ausencia de filtraciones y consistencia de métricas. Esta verificación permite detectar errores de preparación o exportación; no sustituye la validación con datos industriales externos.
'''), code('''
releidas = {}
for nombre, tabla in tablas.items():
    releida = pd.read_csv(ruta_export / nombre)
    assert releida.shape == tabla.shape, nombre
    releidas[nombre] = {'filas': len(releida), 'columnas': len(releida.columns)}
ver_pred = pd.read_csv(ruta_export / 'predicciones_prueba.csv')
assert not ver_pred.duplicated(['lote', 'tiempo_h']).any()
assert set(ver_pred.lote) == set(test_ids)
assert ver_pred[['real_g_l', 'predicho_g_l', 'referencia_g_l']].notna().all().all()
np.testing.assert_allclose(evaluar(ver_pred.real_g_l, ver_pred.predicho_g_l, ver_pred.lote)['MAE_lotes'], met_final.MAE_lotes)
assert set(X).isdisjoint(prohibidas)
assert all(not set(a) & set(b) for i, a in enumerate(particiones.values())
           for j, b in enumerate(particiones.values()) if i < j)
(ruta_proyecto / 'verification/verificacion_principal.json').write_text(
    json.dumps({'estado': 'correcto', 'exportaciones': releidas,
                'lotes_prueba': len(test_ids), 'sin_fuga_por_lote': True}, ensure_ascii=False, indent=2), encoding='utf-8')
print('Verificación completada: exportaciones coherentes, lotes separados y métricas reproducidas.')
''')]

for filename, cells in [('Proyecto_5_Sensor_Virtual_Penicilina.ipynb', main)]:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'},
                   'title': filename.removesuffix('.ipynb').replace('_', ' ')}
    nbf.write(nb, ROOT / 'notebooks' / filename)
    print(filename, len(cells), 'celdas')
