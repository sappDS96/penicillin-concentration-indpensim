"""Ejecuta el notebook y conserva un informe breve de verificación."""
from pathlib import Path
import json
import os
import time
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
runtime = root / '.jupyter_runtime'
runtime.mkdir(exist_ok=True)
os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime)
os.environ['IPYTHONDIR'] = str(runtime / 'ipython')
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
path = root / 'notebooks/Proyecto_5_Sensor_Virtual_Penicilina.ipynb'
nb = nbformat.read(path, as_version=4)
start = time.perf_counter()

def progress(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'Celda {cell_index + 1}/{len(nb.cells)}: {cell.source.splitlines()[0][:80]}', flush=True)

client = NotebookClient(nb, timeout=900, kernel_name='python3',
                        resources={'metadata': {'path': str(root / 'notebooks')}},
                        on_cell_start=progress)
try:
    client.execute()
finally:
    nbformat.write(nb, path)
errors = [o for c in nb.cells for o in c.get('outputs', []) if o.output_type == 'error']
report = {'notebook': path.name, 'seconds': round(time.perf_counter() - start, 2),
          'code_cells': sum(c.cell_type == 'code' for c in nb.cells),
          'executed_cells': sum(c.cell_type == 'code' and c.execution_count is not None for c in nb.cells),
          'errors': len(errors)}
(root / 'verification/execution.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
assert not errors
