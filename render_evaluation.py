"""Render one fitted simulation under capture lighting and learned appearance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from MPMAvatar.evaluation_video import encode_comparison

# Template command (gaugar environment, from Gaussian-Garments):
# python render_evaluation.py --data ../data/GaussianGarments/ClothTransformer/sim_00000 --output output/ClothTransformer/sim_00000 --evaluation output/ClothTransformer/sim_00000/evaluation/held_out --appearance-checkpoint output/ClothTransformer/sim_00000/stage3/epoch5/net.pt --branch gt-lighting
# Repeat with --branch textures, then --branch fitted-appearance. run.py does all three.


def read_json(path: Path) -> dict[str, Any]:
    assert path.is_file(), path
    return json.loads(path.read_text())


def split_frame(data: Path, manifest: dict[str, Any], frame: int) -> tuple[Path, int]:
    for split in manifest['splits'].values():
        if frame in split['global_frame_ids']:
            index = split['global_frame_ids'].index(frame)
            return data / split['directory'], split['local_frame_ids'][index]
    raise ValueError(f'Frame {frame} is outside the exported splits.')


def finish_branch(data: Path, evaluation: Path, branch: str, manifest: dict[str, Any],
                  frames: list[int], details: dict[str, Any]) -> None:
    for camera in manifest['camera_ids']:
        destination = evaluation / branch / camera
        (destination / 'gt').mkdir()
        for frame in frames:
            reference = data / 'capture/rgbs' / camera / f'{camera}_rgb{frame:06d}.png'
            assert reference.is_file(), reference
            shutil.copyfile(reference, destination / 'gt' / f'{frame:04d}.png')
        encode_comparison(destination, frames, manifest['fps'])
    (evaluation / branch / 'manifest.json').write_text(json.dumps({
        'status': 'complete', 'frame_ids': frames, 'camera_ids': manifest['camera_ids'],
        'fps': manifest['fps'], 'geometry': str(evaluation / 'predictions.npz'),
        'reference': 'Original capture RGB', 'video_layout': 'prediction left, reference right',
        **details,
    }, indent=2) + '\n')


def gt_lighting(data: Path, evaluation: Path, manifest: dict[str, Any],
                frames: list[int], vertices: np.ndarray, faces: np.ndarray) -> None:
    import bpy
    from cape_avatar import render_utils as render
    from MPMAvatar.render_gt_lighting import capture_cameras

    capture = read_json(data / 'capture/manifest.json')
    assert capture['status'] == 'complete'
    settings = capture['settings']
    info = read_json(data / 'capture/cam_info.json')
    assert list(info) == manifest['camera_ids']
    with np.load(data / 'preparation/sequence.npz', allow_pickle=False) as source:
        bodies, body_faces = source['body_vertices'], source['body_faces']
        if manifest['component'] == 'clothtransformer_gaussian_garments_export':
            body_name, cloth_name = 'body', 'cloth'
            colors = [settings['materials'][name]['color_linear_rgb'] for name in (body_name, cloth_name)]
        else:
            assert manifest['component'] == 'dgarments_gaussian_garments_export'
            body_name, cloth_name = 'skin', 'dress'
            colors = [source['face_colors'][source['face_labels'] == index][0] for index in (0, 1)]
    assert not (evaluation / 'gt_lighting').exists()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    devices = render.configure_scene(scene, settings)
    scene.render.fps = manifest['fps']
    objects = []
    for name, points, triangles, color in zip(
            (body_name, cloth_name), (bodies[frames[0]], vertices[0]), (body_faces, faces), colors, strict=True):
        material = render.make_material(name, np.asarray(color), settings['materials'][name])
        objects.append(render.create_mesh_object(name, render.scientific_points_to_blender(points), triangles, material))
    cameras = capture_cameras(scene, settings, info, manifest['component'])
    render.create_lights(scene, settings['lights'])
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layers = tree.nodes.new('CompositorNodeRLayers')
    composite = tree.nodes.new('CompositorNodeComposite')
    tree.links.new(layers.outputs['Image'], composite.inputs['Image'])
    for camera in cameras:
        (evaluation / 'gt_lighting' / camera / 'pred').mkdir(parents=True)
        (evaluation / 'body_plate' / camera).mkdir(parents=True)
    for index, frame in enumerate(frames):
        render.update_vertices({body_name: objects[0]}, render.scientific_points_to_blender(bodies[frame]))
        render.update_vertices({cloth_name: objects[1]}, render.scientific_points_to_blender(vertices[index]))
        for offset, (name, camera) in enumerate(cameras.items()):
            scene.camera = camera
            scene.cycles.seed = settings['capture']['seed'] + frame * 8 + offset
            objects[1].visible_camera = True
            scene.render.filepath = str(evaluation / 'gt_lighting' / name / 'pred' / f'{frame:04d}.png')
            bpy.ops.render.render(write_still=True)
            render.assert_rgb_png(Path(scene.render.filepath), info[name]['W'], info[name]['H'])
            # Keep predicted cloth shadows on the body while hiding its RGB.
            objects[1].visible_camera = False
            scene.render.filepath = str(evaluation / 'body_plate' / name / f'{frame:04d}.png')
            bpy.ops.render.render(write_still=True)
    finish_branch(data, evaluation, 'gt_lighting', manifest, frames, {
        'appearance': 'Saved capture materials, lighting and color management', 'devices': devices,
        'body': 'Original source body surface; predicted garment topology',
    })


def bake_textures(data: Path, output: Path, evaluation: Path, manifest: dict[str, Any],
                  frames: list[int], vertices: np.ndarray, faces: np.ndarray) -> None:
    # Load Blender's Embree before native scene imports load Open3D/PyMeshLab.
    import bpy
    from scene.dataloader import AvatarDataloader
    from utils.io_utils import read_obj, write_obj
    template = read_obj(output / 'stage1/template_uv.obj')
    np.testing.assert_array_equal(template['faces'], faces)
    assert len(template['vertices']) == vertices.shape[1]
    meshes = evaluation / 'meshes'
    meshes.mkdir()
    baker = SimpleNamespace(texture_margin=5)
    for frame, points in zip(frames, vertices, strict=True):
        mesh = meshes / f'{frame:05d}.obj'
        write_obj(dict(template, vertices=points), mesh)
        split, local = split_frame(data, manifest, frame)
        body_directory = 'body_mesh' if manifest.get('body_model', 'smplx') == 'raw' else 'smplx'
        body = split / body_directory / f'{local:05d}.ply'
        assert body.is_file(), body
        # Use the same native bake settings/collider as appearance training.
        AvatarDataloader.bake_texture(baker, mesh, body, w=512, h=512, save=True)


def body_occlusion(body: np.ndarray, body_faces: np.ndarray, cloth: np.ndarray,
                   cloth_faces: np.ndarray, calibration: dict[str, Any], width: int,
                   height: int) -> np.ndarray:
    """Match native inference's mesh-depth body/garment occlusion rule."""
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    for vertices, faces in ((body, body_faces), (cloth, cloth_faces)):
        scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)),
                            o3d.core.Tensor(faces.astype(np.uint32)))
    extrinsic = np.eye(4)
    extrinsic[:3] = calibration['extrinsics']
    rays = scene.create_rays_pinhole(o3d.core.Tensor(np.asarray(calibration['intrinsics'])),
                                    o3d.core.Tensor(extrinsic), width, height)
    return scene.cast_rays(rays)['geometry_ids'].numpy() == 0


def fitted_appearance(data: Path, output: Path, evaluation: Path, checkpoint: Path,
                      manifest: dict[str, Any], frames: list[int], vertices: np.ndarray,
                      faces: np.ndarray) -> None:
    import torch
    from gaussian_renderer import render
    from scene.avatar_gaussian_model import AvatarSimulationModel
    from scene.avatar_net import AvatarNet
    from scene.cameras import get_cam_info

    assert not (evaluation / 'fitted_appearance').exists()
    assert checkpoint.is_file(), checkpoint
    torch.manual_seed(0)
    gaussian = AvatarSimulationModel(output, texture_size=512)
    np.testing.assert_array_equal(gaussian.mesh.f.cpu().numpy(), faces)
    args = SimpleNamespace(debug=False, texture_size=512, sh_degree=3,
                           compute_cov3D_python=False, convert_SHs_python=False)
    avatar = AvatarNet(args, gaussian).cuda().eval()
    state = torch.load(checkpoint, map_location='cuda', weights_only=False)
    avatar.load_state_dict(state['avatar_net'])
    gaussian.active_sh_degree = state['activate_sh_degree']
    with np.load(data / 'preparation/sequence.npz', allow_pickle=False) as source:
        bodies, body_faces = source['body_vertices'], source['body_faces']
    info = read_json(data / 'capture/cam_info.json')
    calibration = read_json(data / manifest['splits']['train']['directory'] / 'cameras.json')
    cameras = {name: get_cam_info(calibration[name], h=info[name]['H'], w=info[name]['W'])
               for name in manifest['camera_ids']}
    for name in cameras:
        (evaluation / 'fitted_appearance' / name / 'pred').mkdir(parents=True)
    with torch.no_grad():
        for frame, points in zip(frames, vertices, strict=True):
            gaussian.mesh.v = torch.as_tensor(points, dtype=torch.float32, device='cuda')
            gaussian.update_face_coor()
            ambient = np.asarray(Image.open(evaluation / 'texture/ambient' / f'{frame:05d}.png')) / 255.
            normal = np.asarray(Image.open(evaluation / 'texture/normal' / f'{frame:05d}.png')) / 255.
            ambient_tensor = torch.as_tensor(ambient, dtype=torch.float32, device='cuda')[None]
            normal_tensor = torch.as_tensor(normal, dtype=torch.float32, device='cuda').permute(2, 0, 1)
            for name, camera in cameras.items():
                _, visible = avatar(ambient_tensor, normal_tensor, camera)
                rendered = render(camera, gaussian, args, torch.zeros(3, device='cuda'), vis_mask=visible)
                rgb = rendered['render'].permute(1, 2, 0).cpu().numpy()
                alpha = rendered['alpha'].permute(1, 2, 0).cpu().numpy()
                occluded = body_occlusion(bodies[frame], body_faces, points, faces, calibration[name],
                                          info[name]['W'], info[name]['H'])
                rgb[occluded], alpha[occluded] = 0, 0
                plate = np.asarray(Image.open(evaluation / 'body_plate' / name / f'{frame:04d}.png').convert('RGB')) / 255.
                image = np.clip(rgb + plate * (1 - alpha), 0, 1)
                assert np.isfinite(image).all()
                Image.fromarray(np.round(image * 255).astype(np.uint8)).save(
                    evaluation / 'fitted_appearance' / name / 'pred' / f'{frame:04d}.png')
    finish_branch(data, evaluation, 'fitted_appearance', manifest, frames, {
        'checkpoint': str(checkpoint), 'epoch': state['epoch'],
        'appearance': 'Native fitted Gaussians and view-conditioned appearance network; predicted-mesh AO/normals',
        'body': 'Source body rendered with capture lighting and predicted garment shadows; mesh-depth occlusion',
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'output', 'evaluation', 'appearance-checkpoint'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--branch', choices=('gt-lighting', 'textures', 'fitted-appearance'), required=True)
    args = parser.parse_args()
    data, output, evaluation = args.data.resolve(), args.output.resolve(), args.evaluation.resolve()
    manifest = read_json(data / 'manifest.json')
    rollout = read_json(evaluation / 'rollout.json')
    assert rollout['data'] == str(data)
    with np.load(evaluation / 'predictions.npz', allow_pickle=False) as prediction:
        frames, vertices, faces = prediction['frame_ids'].tolist(), prediction['vertices'], prediction['faces']
    assert frames in (manifest['evaluation_frame_ids'], manifest['frame_ids'])
    assert np.isfinite(vertices).all()
    if args.branch == 'gt-lighting':
        gt_lighting(data, evaluation, manifest, frames, vertices, faces)
    elif args.branch == 'textures':
        bake_textures(data, output, evaluation, manifest, frames, vertices, faces)
    else:
        fitted_appearance(data, output, evaluation, args.appearance_checkpoint.resolve(), manifest, frames, vertices, faces)


if __name__ == '__main__':
    main()
