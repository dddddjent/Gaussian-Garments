"""Regression checks for stale AO, failed bakes, and concurrent frame requests."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import unittest

from utils.texture_cache import texture_cache

# Template command (gaugar environment, from Gaussian-Garments):
# python -m unittest discover -s tests -p test_texture_cache.py -v


class TextureCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.mesh = self.root / 'stage2/train/meshes/frame_00000.obj'
        self.body = self.root / 'input/train/body_mesh/00000.ply'
        for path in (self.mesh, self.body):
            path.parent.mkdir(parents=True)
            path.write_text('initial geometry')
        self.settings = {'texture_size': 32, 'texture_margin': 1}
        self.record = self.mesh.parents[1] / 'texture/cache/frame_00000.json'

    def request(self) -> bool:
        with texture_cache(self.mesh, self.body, self.settings) as (ambient, normal, reusable):
            if not reusable:
                for path in (ambient, normal):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'baked image')
            return reusable

    def test_existing_maps_without_metadata_are_rebaked_once(self) -> None:
        for name in ('ambient', 'normal'):
            path = self.mesh.parents[1] / 'texture' / name / f'{self.mesh.stem}.png'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'old image')
        self.assertFalse(self.request())
        self.assertTrue(self.request())
        self.assertEqual(json.loads(self.record.read_text())['body_model'], 'raw')

    def test_body_and_garment_edits_even_with_preserved_mtime(self) -> None:
        self.assertFalse(self.request())
        for path in (self.body, self.mesh):
            with self.subTest(path=path):
                stat = path.stat()
                path.write_text('changed geometry')  # Same byte count.
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                self.assertFalse(self.request())
                self.assertTrue(self.request())

    def test_settings_body_selection_and_missing_map(self) -> None:
        self.assertFalse(self.request())
        for name, value in (('texture_size', 64), ('texture_margin', 2), ('samples', 16)):
            self.settings[name] = value
            self.assertFalse(self.request())
            self.assertTrue(self.request())
        raw = self.body
        self.body = raw.parent.parent / 'smplx/00000.ply'
        self.body.parent.mkdir()
        self.body.write_bytes(raw.read_bytes())
        self.assertFalse(self.request())
        self.assertEqual(json.loads(self.record.read_text())['body_model'], 'smplx')
        self.body = raw
        self.assertFalse(self.request())
        self.assertEqual(json.loads(self.record.read_text())['body']['path'], str(raw.resolve()))
        normal = self.mesh.parents[1] / 'texture/normal' / f'{self.mesh.stem}.png'
        normal.unlink()
        self.assertFalse(self.request())

    def test_failed_bake_and_missing_body_cannot_reuse_old_maps(self) -> None:
        self.assertFalse(self.request())
        self.body.write_text('new body')
        with self.assertRaisesRegex(RuntimeError, 'bake failed'):
            with texture_cache(self.mesh, self.body, self.settings) as (_, _, reusable):
                self.assertFalse(reusable)
                raise RuntimeError('bake failed')
        self.assertFalse(self.record.exists())
        self.assertFalse(self.request())
        self.body.unlink()
        with self.assertRaises(AssertionError):
            self.request()

    def test_inputs_changed_during_bake_are_rejected(self) -> None:
        with self.assertRaisesRegex(AssertionError, 'Geometry changed'):
            with texture_cache(self.mesh, self.body, self.settings) as (ambient, normal, _):
                for path in (ambient, normal):
                    path.parent.mkdir(parents=True)
                    path.write_bytes(b'baked image')
                self.mesh.write_text('changed mid-bake')
        self.assertFalse(self.record.exists())

    def test_concurrent_requests_bake_only_once(self) -> None:
        with ThreadPoolExecutor(max_workers=8) as workers:
            results = list(workers.map(lambda _: self.request(), range(16)))
        self.assertEqual(results.count(False), 1)
        self.assertEqual(results.count(True), 15)


if __name__ == '__main__':
    unittest.main()
