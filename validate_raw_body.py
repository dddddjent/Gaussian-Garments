"""Bounded raw-body loader, collision-gradient and AO checks, without reconstruction."""

import argparse
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any

# Template commands (activate gaugar; run from Gaussian-Garments):
# python validate_raw_body.py --data ../data/GaussianGarments/ClothTransformer/sim_00000_raw --output ../data/outputs/gaussian_garments_raw_validation --stage collision
# python validate_raw_body.py --data ../data/GaussianGarments/ClothTransformer/sim_00000_raw --output ../data/outputs/gaussian_garments_raw_validation --stage appearance
# Run separately: native registration and Blender use conflicting Embree libraries.


def check_collision(root: Path, work: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    from scene.dataset_readers import Dataloader
    from scene.mesh_model import MeshModel
    from utils.defaults import DEFAULTS
    from utils.preprocess_utils import PrepareDataset

    with np.load(root / manifest['collider_sequence'], allow_pickle=False) as motion:
        vertices, faces = motion['vertices'], motion['faces']
    report: dict[str, Any] = {'splits': {}}
    for split, entry in manifest['splits'].items():
        sequence = root / entry['directory']
        DEFAULTS.data_root = str(sequence.parent.parent)
        loader = Dataloader(SimpleNamespace(subject=manifest['subject'], sequence=sequence.name,
                                           subject_out=str(work), white_background=False))
        assert len(loader.cam_paths) == len(manifest['camera_ids']) == 8
        assert len(loader) == len(entry['local_frame_ids'])
        for local, global_frame in zip(entry['local_frame_ids'], entry['global_frame_ids']):
            body = loader.load_body(local)
            np.testing.assert_array_equal(np.asarray(body.vertices), vertices[global_frame])
            np.testing.assert_array_equal(np.asarray(body.triangles), faces)
        gradients = []
        for frame in (0, len(loader) - 1):
            loader.load_frame(frame)
            assert len(loader.cam_info) == 8
            body = loader.load_body(frame)
            body.compute_triangle_normals()
            normals = np.asarray(body.triangle_normals)[:8]
            centres = np.asarray(body.vertices)[np.asarray(body.triangles)[:8]].mean(1)
            # Independent probe points 2 mm inside known faces; not a garment template.
            model = MeshModel.__new__(MeshModel)
            model.v = torch.tensor(centres - .002 * normals, device='cuda', requires_grad=True)
            model.collision_faces_ids = np.arange(8)[:, None]
            model.init_body(body)
            np.testing.assert_array_equal(model.body.faces, faces)
            loss = model.collision()
            loss.backward()
            assert torch.isfinite(model.v.grad).all() and torch.count_nonzero(model.v.grad) > 0
            assert loss.item() > 0
            gradients.append(float(model.v.grad.norm()))
        report['splits'][split] = {'frames_checked': len(loader), 'camera_count': len(loader.cam_paths),
                                   'gradient_norm_first_last': gradients}
    sequence = root / manifest['splits']['train']['directory']
    PrepareDataset(sequence, work / 'prepared', 'PINHOLE', 0)
    assert len(list((work / 'prepared/images').glob('*.png'))) == 8
    assert len(list((work / 'prepared/masks').glob('*.png'))) == 8
    report.update(body_vertices=vertices.shape[1], body_faces=len(faces),
                  native_stage1_preparation='passed', hand_vertices_removed=0,
                  collision_forward_backward='passed')
    return report


def check_appearance(root: Path, work: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    import bpy
    import numpy as np
    import torch
    from scene.dataloader import AvatarDataloader
    from utils.defaults import DEFAULTS

    # A tiny independent UV triangle exercises texture baking without substituting
    # source garment geometry for the native reconstruction.
    mesh_text = 'v -0.1 1.0 0.3\nv 0.1 1.0 0.3\nv 0.0 1.2 0.3\nvt 0.1 0.1\nvt 0.9 0.1\nvt 0.5 0.9\nf 1/1 2/2 3/3\n'
    (work / 'stage1').mkdir()
    (work / 'stage1/template_uv.obj').write_text(mesh_text)
    preferences = bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type = 'CUDA'
    preferences.get_devices()
    assert any(device.type == 'CUDA' for device in preferences.devices)
    for device in preferences.devices:
        device.use = device.type == 'CUDA'
    bpy.context.scene.cycles.samples = 8
    report: dict[str, Any] = {}
    for split, entry in manifest['splits'].items():
        sequence = root / entry['directory']
        DEFAULTS.data_root = str(sequence.parent.parent)
        args = SimpleNamespace(subject=manifest['subject'], subject_out=work,
                               white_background=False, random_bg=False, blur_mask=False,
                               texture_size=32, texture_margin=1, eval=False, shuffle=False)
        loader = AvatarDataloader(args)
        assert set(loader.dataset_info) == {sequence.name}
        assert len(loader) == len(entry['local_frame_ids']) * 8
        info = loader.dataset_info[sequence.name]
        assert all(path.parent.name == 'body_mesh' for path in info['body_mesh_paths'])
        mesh_dir = work / 'stage2' / sequence.name / 'meshes'
        mesh_dir.mkdir(parents=True)
        for frame in (0, len(entry['local_frame_ids']) - 1):
            mesh = mesh_dir / f'frame_{frame:05d}.obj'
            mesh.write_text(mesh_text)
            data = loader.load_frame(sequence.name, frame, manifest['camera_ids'][0])
            assert data['ambient'].shape == (1, 32, 32)
            assert data['normal'].shape == (3, 32, 32)
            assert torch.isfinite(data['ambient']).all() and torch.isfinite(data['normal']).all()
            objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
            assert len(objects) == 2
            body = next(obj for obj in objects if len(obj.data.vertices) == manifest['collider']['vertex_count'])
            garment = next(obj for obj in objects if len(obj.data.vertices) == 3)
            np.testing.assert_allclose(np.asarray(body.matrix_world), np.asarray(garment.matrix_world), atol=1e-6)
            record = mesh.parents[1] / 'texture/cache' / f'{mesh.stem}.json'
            cache = json.loads(record.read_text())
            assert cache['body_model'] == 'raw'
            assert cache['body']['path'] == str(info['body_mesh_paths'][frame].resolve())
            saved_at = record.stat().st_mtime_ns
            repeated = loader.load_frame(sequence.name, frame, manifest['camera_ids'][0])
            assert record.stat().st_mtime_ns == saved_at, 'Unchanged frame unexpectedly rebaked'
            assert torch.equal(data['ambient'], repeated['ambient'])
            assert torch.equal(data['normal'], repeated['normal'])
        report[split] = {'samples': len(loader), 'baked_local_frames': [0, len(entry['local_frame_ids']) - 1],
                         'body_present_in_bake': True, 'texture_size': 32, 'cycles_samples': 8,
                         'raw_body_cache_record': 'passed', 'cache_reuse_matches_first_load': 'passed'}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('collision', 'appearance'), required=True)
    args = parser.parse_args()
    root, output = args.data.resolve(), args.output.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['body_model'] == 'raw'
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / f'{args.stage}.json'
    assert not report_path.exists(), report_path
    with tempfile.TemporaryDirectory(prefix='gaugar_raw_validation_') as directory:
        work = Path(directory)
        os.environ['GAUGAR_DATA_ROOT'] = str(root / 'input')
        os.environ['GAUGAR_OUTPUT_ROOT'] = str(work)
        os.environ['GAUGAR_AUX_ROOT'] = str(work / 'absent_auxiliary')
        check = check_collision if args.stage == 'collision' else check_appearance
        results = check(root, work, manifest)
    report_path.write_text(json.dumps({'status': 'passed', 'dataset': str(root), 'results': results,
        'scope': 'Bounded input/collision/AO checks only; no reconstructed garment, registration training, appearance training or physical prediction',
        'temporary_outputs': 'removed'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
