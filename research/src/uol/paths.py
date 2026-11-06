"""Project locations and explicit aliases for paths recorded before reorganization.

Only paths inside this project, or relative project paths, are translated.
Source files and frozen manifests retain their original bytes. Their old path
strings resolve through the recorded migration map, not through replacement files.
"""
from functools import lru_cache
import hashlib
import json
import posixpath
from pathlib import Path as NativePath

_ROOT = NativePath(__file__).resolve().parents[2]
_ConcretePath = type(_ROOT)

@lru_cache(maxsize=1)
def aliases():
    path = _ROOT / 'research/paths.json'
    if not path.exists():
        path = _ROOT / 'provenance/reorganization/moves.json'
    mapping = json.loads(path.read_text(encoding='utf-8-sig'))['aliases']
    # A path can be joined a segment at a time. Register the intermediate
    # spellings too, e.g. research/programs/next_phase/state.json.
    expanded = dict(mapping)
    for old, new in mapping.items():
        for parent, target in mapping.items():
            if old.startswith(parent + '/'):
                expanded[target + old[len(parent):]] = new
    return sorted(expanded.items(), key=lambda pair: len(pair[0]), reverse=True)

@lru_cache(maxsize=8192)
def _canonical(path):
    path = NativePath(path)
    absolute = path.is_absolute()
    if absolute:
        try:
            relative = path.relative_to(_ROOT)
        except ValueError:
            return path
    else:
        relative = path
    value = relative.as_posix()
    # Reject traversal before translation. External relative paths stay native.
    if '..' in relative.parts:
        return path
    for old, new in aliases():
        if value.casefold() == old.casefold() or value.casefold().startswith(old.casefold() + '/'):
            mapped = NativePath(new + value[len(old):])
            return _ROOT / mapped if absolute else mapped
    return path

class ProjectPath(_ConcretePath):
    """A pathlib path that also accepts recorded project aliases.

    pathlib's segment operations construct this subclass, so joins, parents,
    with_name, absolute paths and relative paths use the same translation.
    Native paths outside the repository are left unchanged.
    """
    def __init__(self, *paths):
        super().__init__(_canonical(NativePath(*paths)))

def project_root():
    return ProjectPath(_ROOT)

def resolve_path(path):
    result = ProjectPath(path)
    return result if result.is_absolute() else project_root() / result

@lru_cache(maxsize=8192)
def resolve_document_link(document, target):
    """Resolve a retained document's link against its original project location."""
    current = NativePath(document).resolve()
    target = target.replace('\\', '/')
    direct = NativePath(current.parent / target).resolve()
    if direct.exists():
        return ProjectPath(direct)
    relative = current.relative_to(_ROOT).as_posix()
    original = original_locations().get(relative, relative)
    old_target = posixpath.normpath(posixpath.join(posixpath.dirname(original), target))
    resolved = resolve_path(old_target)
    if resolved.exists():
        return resolved
    # Prior-dossier documents were copied from the old dossier root without
    # changing their relative links. Preserve that recorded source context.
    if '/prior_dossier/' in original:
        copied_target = resolve_path(posixpath.normpath(posixpath.join('dossier', target)))
        if copied_target.exists():
            return copied_target
    return resolve_path(target)

@lru_cache(maxsize=1)
def original_locations():
    plan = json.loads((_ROOT / 'provenance/reorganization/moves.json').read_text(encoding='utf-8'))
    return {row['new']: row['old'] for row in plan['files']}

def historical_source(path, expected_sha256):
    """Find the exact source bytes recorded by a saved experiment."""
    candidate = resolve_path(path)
    if candidate.is_file() and hashlib.sha256(candidate.read_bytes()).hexdigest() == expected_sha256:
        return candidate
    baseline = json.loads((_ROOT / 'provenance/reorganization/baseline.json').read_text(encoding='utf-8'))
    logical = NativePath(path).as_posix()
    for item in baseline['files']:
        if (item['path'] == logical or resolve_path(item['path']) == candidate) and item['sha256'] == expected_sha256:
            preserved = NativePath(_ROOT / item['snapshot'])
            if hashlib.sha256(preserved.read_bytes()).hexdigest() != expected_sha256:
                raise ValueError(f'Preserved source hash mismatch: {preserved}')
            return preserved
    raise FileNotFoundError(f'No verified historical source for {logical} with SHA-256 {expected_sha256}')

def legacy_output(filename, module):
    """Route historical narrative generators away from the current entry points.

    Existing generators expect the old README sections and HANDOFF fields.
    Their results belong to that report's history. Creating this copy happens
    only when the generator is explicitly run, never during import.
    """
    directory = _ROOT / 'research/archive/generated' / module.rsplit('.', 1)[-1]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if not target.exists():
        original = _ROOT / 'provenance/snapshots/pre_reorganization_20260912' / filename
        target.write_bytes(original.read_bytes())
    return ProjectPath(target)
