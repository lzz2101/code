"""Saved-bounds I/O recovery tests; no pilot or simulator is executed."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mira_leo_study.config import load_config
from mira_leo_study.errors import ArtifactReadError
from mira_leo_study.paper_results import run_paper_results
from mira_leo_study.reward_pilot import load_reward_bounds_file, reward_context


class RewardBoundsIOTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).parents[1] / 'configs/base.yaml')
        self.payload = dict(phi_theta_deg=15., B_penalty=.05, reward_context=reward_context(self.config, 0),
                            reward_identity='artificial-io-fixture', reward_lower=-1000., reward_upper=1000.)

    def test_valid_bom_file_keeps_frozen_values_and_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bounds.json'
            path.write_text(json.dumps(self.payload), encoding='utf-8-sig')
            result = load_reward_bounds_file(self.config, path, 0)
            self.assertEqual(result.get('algorithm.B_penalty'), .05)
            self.assertEqual(result.get('algorithm.reward_bounds.identity'), 'artificial-io-fixture')
            self.assertEqual(result.get('algorithm.reward_bounds.lower'), -1000.)
            self.assertIsNone(self.config.get('algorithm.reward_bounds.lower'))

    def test_cloud_errno22_has_actionable_message(self):
        with patch.object(Path, 'read_text', side_effect=OSError(22, 'Invalid argument')):
            with self.assertRaises(ArtifactReadError) as raised:
                load_reward_bounds_file(self.config, Path('OneDrive/outputs/reward_bounds.json'), 0)
        message = str(raised.exception)
        for fragment in ('OneDrive', '--reward-bounds', '--outputs', 'No automatic recalibration'):
            self.assertIn(fragment, message)

    def test_missing_file_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ArtifactReadError, 'Cannot read reward-bounds'):
                load_reward_bounds_file(self.config, Path(temp) / 'missing.json', 0)

    def test_malformed_json_is_not_recalibrated(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bounds.json'; path.write_text('{incomplete', encoding='utf-8')
            with self.assertRaisesRegex(ArtifactReadError, 'Invalid reward-bounds JSON'):
                load_reward_bounds_file(self.config, path, 0)

    def test_json_array_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bounds.json'; path.write_text('[]', encoding='utf-8')
            with self.assertRaisesRegex(ArtifactReadError, 'must be an object'):
                load_reward_bounds_file(self.config, path, 0)

    def test_context_validation_is_not_bypassed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bounds.json'
            path.write_text(json.dumps({**self.payload, 'reward_context':'wrong-context'}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'different configuration/seed'):
                load_reward_bounds_file(self.config, path, 0)

    def test_unreadable_cache_stops_before_pilot_and_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); cache = root / 'reward_bounds/reward_bounds.json'
            cache.parent.mkdir(); cache.write_text('{}', encoding='utf-8')
            with patch.object(Path, 'read_text', side_effect=OSError(22, 'Invalid argument')), \
                 patch('mira_leo_study.paper_results.run_reward_bounds_pilot') as pilot, \
                 patch('mira_leo_study.paper_results.run_simulation') as simulation:
                with self.assertRaises(ArtifactReadError):
                    run_paper_results(self.config, root, diagnostics=True)
                pilot.assert_not_called(); simulation.assert_not_called()
            self.assertFalse((root / 'summary.json').exists())
            self.assertEqual(cache.read_text(encoding='utf-8'), '{}')

    def test_cli_prints_message_without_python_traceback(self):
        from mira_leo_study.__main__ import main
        captured = io.StringIO()
        with patch('mira_leo_study.__main__.run_paper_results', side_effect=ArtifactReadError('artificial unreadable calibration')), \
             contextlib.redirect_stderr(captured):
            with self.assertRaises(SystemExit) as raised:
                main(['paper-results', '--diagnostics'])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('artificial unreadable calibration', captured.getvalue())
        self.assertNotIn('Traceback', captured.getvalue())


if __name__ == '__main__':
    unittest.main()
