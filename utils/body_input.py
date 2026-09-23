"""Shared camera and body discovery for native SMPL-X and raw mesh sequences."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import open3d as o3d


def camera_directories(sequence: Path) -> list[Path]:
    """Use calibrated cameras, excluding body and other non-camera directories."""
    cameras = json.loads((sequence / 'cameras.json').read_text())
    assert cameras, f'No calibrated cameras: {sequence}'
    paths = [sequence / name for name in sorted(cameras)]
    for path in paths:
        assert path.is_dir(), path
    return paths


def body_directory(sequence: Path) -> Path:
    """Require exactly one explicit body layout; never substitute a fitted body."""
    paths = [sequence / name for name in ('smplx', 'body_mesh') if (sequence / name).is_dir()]
    assert len(paths) == 1, f'Expected exactly one smplx/ or body_mesh/ directory: {sequence}'
    return paths[0]


def body_mesh_paths(sequence: Path, frame_count: int) -> list[Path]:
    directory = body_directory(sequence)
    paths = sorted(directory.glob('*.ply'))
    assert len(paths) == frame_count, f'Body/image frame count mismatch: {directory}'
    assert [path.stem for path in paths] == [f'{frame:05d}' for frame in range(frame_count)], directory
    return paths


def load_collision_body(path: Path, aux_root: Path) -> 'o3d.geometry.TriangleMesh':
    """Keep raw topology intact; apply the native hand exclusion only to SMPL-X."""
    import open3d as o3d

    assert path.is_file(), path
    assert path.parent.name in ('smplx', 'body_mesh'), path
    body = o3d.io.read_triangle_mesh(str(path))
    assert body.has_vertices() and body.has_triangles(), path
    if path.parent.name == 'smplx':
        segmentation = aux_root / 'smplx/smplx_vert_segmentation.json'
        assert segmentation.is_file(), segmentation
        labels = json.loads(segmentation.read_text())
        hands = [vertex for name, vertices in labels.items() for vertex in vertices if 'hand' in name.lower()]
        assert hands and min(hands) >= 0 and max(hands) < len(body.vertices)
        body.remove_vertices_by_index(hands)
    return body
