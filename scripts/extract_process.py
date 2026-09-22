"""Extrae las primeras 37 posiciones de IndPenSim sin alterar filas ni valores.

Uso: python scripts/extract_process.py /ruta/a/Mendeley_data
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source_dir', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = root / 'data/source'
out.mkdir(parents=True, exist_ok=True)
source = args.source_dir / '100_Batches_IndPenSim_V3.csv'
summary = args.source_dir / '100_Batches_IndPenSim_Statistics.csv'
df = pd.read_csv(source, usecols=range(37))
if df.shape != (113935, 37):
    raise ValueError(f'La versión tiene dimensiones distintas: {df.shape}; revisar el esquema antes de continuar.')
df.to_csv(out / 'indpensim_process_original.csv.gz', index=False,
          compression={'method': 'gzip', 'mtime': 0})
shutil.copy2(summary, out / summary.name)
digest = hashlib.sha256()
with source.open('rb') as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        digest.update(block)
metadata = {'source_file': source.name, 'source_bytes': source.stat().st_size,
            'source_sha256': digest.hexdigest(), 'rows': len(df),
            'extracted_positions_zero_based': list(range(37)),
            'doi': '10.17632/pdnjz7zz5x.2', 'license': 'CC BY 4.0',
            'source_url': 'https://data.mendeley.com/datasets/pdnjz7zz5x/2',
            'note': 'Extracto sin imputación ni filtrado de filas; encabezados originales en las primeras 37 posiciones. Raman no incluido.'}
(out / 'provenance.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Extracto guardado: {len(df):,} registros y {len(df.columns)} columnas.')
