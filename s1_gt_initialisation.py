"""Optional ClothTransformer GT frame-0 initialization; native COLMAP is unchanged."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection

from utils.io_utils import read_obj, storePly, write_obj

# Template command (activate gaugar; run from Gaussian-Garments):
# python s1_gt_initialisation.py --data ../data/GaussianGarments/ClothTransformer/sim_00000 --output output/ClothTransformer/sim_00000_gt_init --template-frame 0
# Prefer run.py --stage initialize --initialization gt-mesh to record the experiment.


def garment_template(source: dict[str, np.ndarray], body_vertices: int,
                     body_faces: int) -> dict[str, np.ndarray]:
    """Extract cloth while preserving source vertices, face order and UV seams."""
    vertices = source['vertices'][body_vertices:]
    faces = source['faces'][body_faces:] - body_vertices
    assert len(vertices) > 0 and faces.ndim == 2 and faces.shape[1] == 3
    assert source['faces'][:body_faces].max() < body_vertices
    assert faces.min() >= 0 and faces.max() < len(vertices)
    assert np.isfinite(vertices).all()
    uv_faces = source['texture_faces'][body_faces:]
    assert uv_faces.shape == faces.shape and uv_faces.min() >= 0
    used_uv, inverse = np.unique(uv_faces, return_inverse=True)
    uvs = source['uvs'][used_uv]
    assert np.isfinite(uvs).all() and np.all((uvs >= 0) & (uvs <= 1))
    return {'vertices': vertices, 'faces': faces, 'uvs': uvs,
            'texture_faces': inverse.reshape(faces.shape)}


@torch.no_grad()
def observed_point_cloud(source: dict[str, np.ndarray], body_faces: int,
                         train: Path, frame: int
                         ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Backproject visible cloth pixels using the supplied frame's surface.

    Rasterizing the combined original body/cloth preserves body occlusions.
    Only garment-mask pixels contribute observed RGB; no material-color fill.
    """
    assert torch.cuda.is_available(), 'GT point-cloud preparation requires the gaugar CUDA environment.'
    vertices = torch.as_tensor(source['vertices'], dtype=torch.float32, device='cuda')
    faces = torch.as_tensor(source['faces'], dtype=torch.int64, device='cuda')
    mesh = Meshes(verts=[vertices], faces=[faces])
    cameras = json.loads((train / 'cameras.json').read_text())
    points, colors, reports = [], [], {}
    for name, calibration in cameras.items():
        with Image.open(train / name / 'rgb_images' / f'{frame:05d}.png') as image:
            rgb = np.array(image.convert('RGB'))
        with Image.open(train / name / 'garment_masks' / f'{frame:05d}.png') as image:
            mask = np.array(image) > 127
        height, width = mask.shape
        assert rgb.shape == (height, width, 3)
        assert mask.any(), f'No visible garment in {name}, frame {frame}'
        extrinsic = torch.tensor(calibration['extrinsics'], dtype=torch.float32, device='cuda')
        intrinsic = torch.tensor(calibration['intrinsics'], dtype=torch.float32, device='cuda')
        camera = cameras_from_opencv_projection(
            R=extrinsic[None, :, :3], tvec=extrinsic[None, :, 3],
            camera_matrix=intrinsic[None],
            image_size=torch.tensor([[height, width]], dtype=torch.float32, device='cuda'))
        settings = RasterizationSettings(image_size=(height, width), blur_radius=0,
                                         faces_per_pixel=1, perspective_correct=True,
                                         cull_backfaces=False)
        fragments = MeshRasterizer(cameras=camera, raster_settings=settings)(mesh)
        face_ids = fragments.pix_to_face[0, ..., 0]
        observed = torch.as_tensor(mask, device='cuda')
        predicted = face_ids >= body_faces
        valid = observed & predicted
        count = int(valid.sum())
        coverage = count / int(observed.sum())
        iou = count / int((observed | predicted).sum())
        assert coverage > 0.9, f'{name}: source mesh/calibration mask coverage is only {coverage:.3f}'
        barycentric = fragments.bary_coords[0, ..., 0, :][valid]
        xyz = (vertices[faces[face_ids[valid]]] * barycentric[..., None]).sum(dim=1)
        assert torch.isfinite(xyz).all()
        points.append(xyz.cpu().numpy())
        colors.append(rgb[valid.cpu().numpy()])
        reports[name] = {'points': count, 'observed_mask_coverage': coverage,
                         'visible_garment_mask_iou': iou}
        print(f'{name}: {count} colored points, mask coverage {coverage:.3f}, IoU {iou:.3f}', flush=True)
    return np.concatenate(points), np.concatenate(colors), reports


def initialize(root: Path, output: Path, frame: int) -> None:
    """Supply only frame-0 cloth geometry and observed colors to Stage 2."""
    assert frame == 0, 'GT mesh initialization currently supplies frame 0 only.'
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['component'] == 'clothtransformer_gaussian_garments_export'
    split = manifest['splits']['train']
    assert split['local_frame_ids'][frame] == split['global_frame_ids'][frame] == 0
    train = root / split['directory']
    capture = json.loads((root / 'capture/manifest.json').read_text())
    assert capture['status'] == 'complete' and capture['surface_type'] == 'original_body_and_cloth'
    source_path = root / 'capture/uvmesh/mesh_cloth_000000.obj'
    assert source_path.is_file(), source_path
    destination = output / 'stage1'
    assert not destination.exists(), destination
    source = read_obj(source_path)
    template = garment_template(source, capture['body_vertex_count'], capture['body_face_count'])
    points, colors, cameras = observed_point_cloud(source, capture['body_face_count'], train, frame)
    destination.mkdir(parents=True)
    write_obj(template, destination / 'template_uv.obj')
    write_obj({'vertices': template['vertices'], 'faces': template['faces']},
              destination / 'template.obj')
    storePly(destination / 'point_cloud.ply', points, colors)
    report = {
        'status': 'complete', 'initialization': 'gt-mesh', 'dataset': str(root),
        'template_frame': frame, 'source_mesh': str(source_path),
        'source_body_vertices_removed': capture['body_vertex_count'],
        'source_body_faces_removed': capture['body_face_count'],
        'garment_vertices': len(template['vertices']), 'garment_faces': len(template['faces']),
        'uv_coordinates': len(template['uvs']), 'colored_points': len(points), 'cameras': cameras,
        'geometry': 'Original frame-0 garment only; topology and coordinates preserved; no remeshing',
        'uv': 'Existing capture UVs, with unused body UV entries removed',
        'color': 'Observed training-frame-0 RGB at visible garment pixels; source-mesh backprojection',
        'gt_cloth_frames_used': [0], 'heldout_observations_used': False,
        'protocol': 'GT-initialized template; subsequent registration and appearance remain learned',
        'downstream_training': 'unrun',
    }
    (destination / 'initialization.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'GT frame-0 initialization complete: {destination}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--template-frame', type=int, default=0)
    args = parser.parse_args()
    initialize(args.data.resolve(), args.output.resolve(), args.template_frame)


if __name__ == '__main__':
    main()
