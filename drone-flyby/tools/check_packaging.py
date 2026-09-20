"""Static check that the Docker image would contain everything /predict needs.

The Windows Docker daemon is not always available on the development host, and
a packaging mistake is one of the cheapest ways to lose a whole attempt. This
walks the serving import graph and asserts that every module it reaches, and
every asset it opens, is something the Dockerfile actually copies.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENTRY_POINTS = ('api.py', 'example.py')


def copied_paths() -> Set[str]:
    """Everything the Dockerfile's COPY lines bring into the image."""
    text = (ROOT / 'Dockerfile').read_text(encoding='utf-8')
    copied: Set[str] = set()
    for line in text.splitlines():
        match = re.match(r'^\s*COPY\s+(.*)$', line)
        if not match:
            continue
        parts = match.group(1).split()
        for token in parts[:-1]:
            copied.add(token.strip())
    return copied


def local_imports(path: Path, seen: Set[Path]) -> Set[str]:
    """Repo-local modules reachable from a file, transitively."""
    if path in seen or not path.is_file():
        return set()
    seen.add(path)
    modules: Set[str] = set()
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for name in names:
            top = name.split('.')[0]
            candidate_module = ROOT / f'{top}.py'
            candidate_package = ROOT / top / '__init__.py'
            if candidate_module.is_file():
                modules.add(f'{top}.py')
                modules |= local_imports(candidate_module, seen)
            elif candidate_package.is_file():
                modules.add(f'{top}/')
                target = ROOT / Path(*name.split('.'))
                for suffix in (target.with_suffix('.py'), target / '__init__.py'):
                    modules |= local_imports(suffix, seen)
    return modules


def main() -> int:
    copied = copied_paths()
    failures = []

    reachable: Set[str] = set()
    seen: Set[Path] = set()
    for entry in ENTRY_POINTS:
        reachable.add(entry)
        reachable |= local_imports(ROOT / entry, seen)

    print('Serving import graph:')
    for module in sorted(reachable):
        present = module in copied or module.rstrip('/') in copied
        print(f'  {"OK  " if present else "MISS"} {module}')
        if not present:
            failures.append(f'Dockerfile does not COPY {module}')

    print('\nRuntime assets:')
    from v2.config import CONFIG  # noqa: E402  (import after sys.path setup)

    for label, value in (
        ('gallery', CONFIG.recognizer.gallery_path),
        ('proposal weights', CONFIG.proposals.yolo_weights),
    ):
        path = Path(value)
        exists = path.is_file()
        try:
            relative = path.relative_to(ROOT).parts[0]
        except ValueError:
            relative = str(path)
        inside = relative in copied or f'{relative}/' in copied
        size = path.stat().st_size / 1e6 if exists else 0.0
        print(f'  {"OK  " if exists and inside else "MISS"} {label}: {path} '
              f'({size:.2f} MB, copied={inside})')
        if not exists:
            failures.append(f'{label} missing at {path}')
        if not inside:
            failures.append(f'{label} is outside any COPY path')

    print('\nSample data independence:')
    # src/ is 400 MB and deliberately not shipped; nothing in the serving path
    # may depend on it.
    sample_readers = {'load_frame', 'load_annotations', 'frame_numbers', 'load_sample',
                      'scene_directory', 'load_run_metadata'}
    bad = []
    for module in sorted(reachable):
        target = ROOT / module.rstrip('/')
        files = [target] if target.is_file() else sorted(target.rglob('*.py'))
        for file in files:
            tree = ast.parse(file.read_text(encoding='utf-8'), filename=str(file))
            defined = {
                node.name for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for node in ast.walk(tree):
                # A real call site, not the definition of the helper itself.
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, 'id', None) or getattr(node.func, 'attr', None)
                if name in sample_readers and name not in defined:
                    bad.append(f'{file.relative_to(ROOT)}:{node.lineno} calls {name}()')
    if bad:
        for row in bad:
            print(f'  WARN {row}')
    else:
        print('  OK   nothing in the serving path reads src/')

    if failures:
        print('\nFAILED:')
        for row in failures:
            print(f'  - {row}')
        return 1
    print('\nPackaging check passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
