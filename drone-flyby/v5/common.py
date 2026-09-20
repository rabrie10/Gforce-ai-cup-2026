"""Shared artifact contracts; no training labels are imported by inference."""
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np


def require_azure():
    if platform.system() != 'Linux' or os.getenv('V5_AZURE_VM') != '1':
        raise RuntimeError('Model execution requires the explicitly configured Azure Linux runtime')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def timing(values):
    return {k: float(np.percentile(values, q)) for k, q in [('p50', 50), ('p95', 95), ('max', 100)]} if values else {}


def verify_assets(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if sha256(root / name) != expected:
            raise ValueError(f'Artifact hash mismatch: {name}')
    return manifest
