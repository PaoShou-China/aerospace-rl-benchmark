"""Isolated historical benchmark runtime for the landing experiment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ANCHOR = "39e6bb6fd5d0f5432f7405a76946938a469492ce"
DEV_SEEDS = (1000007, 2000007, 3000007, 4000007)
CHECKPOINT = HERE / "assets/phase_aware_feedback_repair.pt"

def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_snapshot(out):
    expected = json.loads((out / 'upstream_manifest.json').read_text())
    for name, digest in expected.items():
        if sha256(out / 'source' / name) != digest:
            raise RuntimeError(f'upstream snapshot changed: {name}')
    return expected

def prepare_snapshot(out):
    out.mkdir(parents=True, exist_ok=False)
    archive = out / 'upstream_source.tar.gz'
    subprocess.run(['git', 'archive', '--format=tar.gz', f'--output={archive}', ANCHOR],
                   cwd=ROOT, check=True)
    source = out / 'source'
    source.mkdir()
    with tarfile.open(archive) as stream:
        stream.extractall(source, filter='data')
    manifest = {str(p.relative_to(source)): sha256(p)
                for p in sorted(source.rglob('*')) if p.is_file()}
    write_json(out / 'upstream_manifest.json', manifest)
    (out / 'launcher.py').write_bytes(Path(__file__).read_bytes())
    return source

def setup_imports(source, out, threads):
    # Refuse mixed modules when called from a non-fresh interpreter.
    if any(n == 'common' or n.startswith('common.') or n.startswith('pyflyt_genesis')
           for n in sys.modules):
        raise RuntimeError('use a fresh Python process for the upstream snapshot')
    for name in ('genesis', 'matplotlib'):
        (out / 'cache' / name).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('GS_CACHE_FILE_PATH', str(out / 'cache' / 'genesis'))
    os.environ.setdefault('MPLCONFIGDIR', str(out / 'cache' / 'matplotlib'))
    sys.path.insert(0, str(source / 'src'))
    import torch
    torch.set_num_threads(threads)
    import common.train_utils as upstream
    assert Path(upstream.__file__).resolve() == source / 'src/common/train_utils.py'
    return torch, upstream
