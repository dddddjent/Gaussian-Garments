"""Evaluation checkpoint selection and split-command integration checks."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation_runner import evaluate, select_appearance, select_checkpoint

# Template command (gaugar environment, from Gaussian-Garments):
# python -m unittest discover -s tests -p test_evaluation.py -v


class EvaluationTests(unittest.TestCase):
    def test_numeric_checkpoints_and_explicit_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / 'stage4/checkpoints'
            checkpoints.mkdir(parents=True)
            for name in ('step_9.pth', 'step_100.pth', 'step_bad.pth'):
                (checkpoints / name).touch()
            self.assertEqual(select_checkpoint(root, None).name, 'step_100.pth')
            self.assertEqual(select_checkpoint(root, checkpoints / 'step_9.pth').name, 'step_9.pth')
            for epoch in (2, 11):
                destination = root / f'stage3/epoch{epoch}'
                destination.mkdir(parents=True)
                (destination / 'net.pt').touch()
            self.assertEqual(select_appearance(root, None).parent.name, 'epoch11')

    def test_missing_fitted_checkpoint_is_not_pretrained_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AssertionError, 'simulation-fit first'):
                select_checkpoint(Path(directory), None)

    def test_both_ranges_share_rollout_and_both_renderers(self) -> None:
        self.check_both_ranges('smplx')

    def test_raw_body_uses_same_continuous_rollout_and_renderers(self) -> None:
        self.check_both_ranges('raw')

    def check_both_ranges(self, body_model: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for file in ('python', 'config.json', 'stage4/finetune.yaml', 'stage4/preparation.json',
                         'stage4/checkpoints/step_2.pth', 'stage3/epoch5/net.pt',
                         'preparation/sequence.npz', 'capture/manifest.json', 'capture/cam_info.json'):
                path = root / file
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (root / 'config.json').write_text(json.dumps({'render_python': str(root / 'python')}))
            args = argparse.Namespace(initialization='colmap', simulation_python=root / 'python',
                                      checkpoint=None, appearance_checkpoint=None, evaluate_from_start=False,
                                      contourcraft_root=Path(__file__).resolve().parents[2] / 'ContourCraft',
                                      aux_root=root, dry_run=False)
            manifest = {'body_model': body_model, 'fps': 25, 'frame_ids': list(range(10)),
                        'evaluation_frame_ids': [8, 9]}
            for full in (False, True):
                args.evaluate_from_start = full
                scope = 'full_sequence' if full else 'held_out'
                # Emulate only the subprocess outputs; inspect the real orchestration.
                def execute(command: list[str], **kwargs: object) -> None:
                    destination = Path(command[command.index('--evaluation') + 1])
                    destination.mkdir(parents=True, exist_ok=True)
                with patch('evaluation_runner.subprocess.run', side_effect=execute) as run:
                    evaluate(args, manifest, root, root)
                commands = [call.args[0] for call in run.call_args_list]
                self.assertEqual(len(commands), 4)
                self.assertEqual('--evaluate-from-start' in commands[0], full)
                self.assertEqual([command[-1] for command in commands[1:]],
                                 ['gt-lighting', 'textures', 'fitted-appearance'])
                result = json.loads((root / 'evaluation' / scope / 'manifest.json').read_text())
                self.assertEqual(result['frame_ids'], list(range(10)) if full else [8, 9])

    def test_unsupported_body_stops_before_launch(self) -> None:
        with patch('evaluation_runner.subprocess.run') as run:
            with self.assertRaisesRegex(AssertionError, 'Unsupported'):
                evaluate(argparse.Namespace(), {'body_model': 'unknown'}, Path('/unused'), Path('/unused'))
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
