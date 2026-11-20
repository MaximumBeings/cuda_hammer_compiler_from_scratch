"""Require explicit, nonempty validation coverage before executing any checks.

This checked-in manifest is updated deliberately when the book's lessons change;
validators never regenerate it from the input tree they are meant to validate.
"""
from pathlib import Path
import json

class CoverageError(RuntimeError):
    pass

def load_manifest(root):
    path=Path(root)/'tools/book-manifest.json'
    if not path.is_file():raise CoverageError('Missing required book manifest: '+str(path))
    try: manifest=json.loads(path.read_text())
    except (OSError,ValueError) as ex:raise CoverageError('Invalid book manifest: '+str(ex)) from ex
    if not isinstance(manifest,dict) or manifest.get('schema')!=1:raise CoverageError('Unsupported book manifest schema')
    for key in ['pages','sources']:
        values=manifest.get(key)
        if not isinstance(values,list) or not values or any(not isinstance(x,str) for x in values) or len(values)!=len(set(values)):
            raise CoverageError('Expected nonempty unique '+key+' in book manifest')
        if any(not isinstance(x,str) or Path(x).is_absolute() or '..' in Path(x).parts for x in values):
            raise CoverageError('Invalid relative path in '+key)
    drivers=manifest.get('gpu_drivers')
    if not isinstance(drivers,dict) or not drivers:raise CoverageError('Expected nonempty GPU driver manifest')
    for path,programs in drivers.items():
        if path not in manifest['sources'] or not path.endswith('.cu') or not isinstance(programs,list) or not programs:
            raise CoverageError('Invalid GPU driver entry: '+str(path))
        if any(not isinstance(p,str) or not p.startswith('/tmp/hammer_ch18_') for p in programs):
            raise CoverageError('Invalid generated executable entry')
    return manifest

def require_inputs(root,kind):
    root=Path(root);manifest=load_manifest(root);docs=root/'docs'
    if not docs.is_dir():raise CoverageError('Missing required docs directory')
    if kind=='host':
        expected={p for p in manifest['sources'] if p.endswith('.cpp')}
        actual={str(p.relative_to(root)) for p in docs.rglob('code/*.cpp') if not p.name.startswith('.')}
    elif kind=='gpu':
        expected=set(manifest['gpu_drivers'])
        actual={str(p.relative_to(root)) for p in (docs/'part4/code').glob('*.cu') if not p.name.startswith('.')}
    elif kind=='pages':
        expected=set(manifest['pages']);actual={str(p.relative_to(root)) for p in docs.rglob('*.md')}
    elif kind=='sources':
        expected=set(manifest['sources']);actual={str(p.relative_to(root)) for p in docs.rglob('code/*') if p.is_file() and not p.name.startswith('.')}
    else:raise CoverageError('Unknown validation input kind: '+kind)
    if not expected or not actual:raise CoverageError('No '+kind+' inputs collected; refusing empty validation')
    missing=expected-actual;extra=actual-expected
    if missing or extra:
        raise CoverageError(kind+' coverage differs from manifest; missing='+repr(sorted(missing))+' extra='+repr(sorted(extra)))
    return [root/p for p in sorted(expected)]
