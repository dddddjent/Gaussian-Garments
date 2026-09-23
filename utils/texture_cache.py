"""Cache baked maps against their actual geometry and bake settings, without hashes."""

from collections.abc import Iterator
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

# Used by appearance fitting (gaugar environment, from Gaussian-Garments):
# python run.py --data ../data/GaussianGarments/ClothTransformer/sim_00000_raw --output output/ClothTransformer/sim_00000_raw --stage appearance --template-frame 0


def file_state(path: Path) -> dict[str, str | int]:
    """Detect replacements and edits, including edits preserving modification time."""
    assert path.is_file(), path
    resolved = path.resolve()
    stat = resolved.stat()
    return {'path': str(resolved), 'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}


@contextmanager
def texture_cache(
    mesh: Path, body: Path, settings: dict[str, object],
) -> Iterator[tuple[Path, Path, bool]]:
    """Lock one frame through baking/loading; publish metadata only after success."""
    assert body.parent.name in ('body_mesh', 'smplx'), body
    directory = mesh.parents[1] / 'texture'
    ambient = directory / 'ambient' / f'{mesh.stem}.png'
    normal = directory / 'normal' / f'{mesh.stem}.png'
    record = directory / 'cache' / f'{mesh.stem}.json'
    record.parent.mkdir(parents=True, exist_ok=True)
    # Separate opens serialize data-loader workers requesting the same frame.
    with record.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        inputs = {'mesh': file_state(mesh), 'body': file_state(body),
                  'body_model': 'raw' if body.parent.name == 'body_mesh' else 'smplx',
                  'settings': settings}
        reusable = (ambient.is_file() and normal.is_file() and record.is_file()
                    and json.loads(record.read_text()) == inputs)
        if not reusable:
            record.unlink(missing_ok=True)
            print(f'Baking AO/normal maps: {mesh.name}, body={body}', flush=True)
        yield ambient, normal, reusable
        if not reusable:
            assert ambient.is_file() and normal.is_file(), 'Bake did not save both texture maps'
            assert file_state(mesh) == inputs['mesh'] and file_state(body) == inputs['body'], (
                'Geometry changed during AO baking; rerun with stable input meshes')
            temporary = record.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(inputs, indent=2) + '\n')
            temporary.replace(record)
