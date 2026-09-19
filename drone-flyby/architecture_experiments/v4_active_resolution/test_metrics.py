import unittest
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from metrics import contains, gate, iou, lift, size_bucket, summarize


class MetricsTests(unittest.TestCase):
    def test_coordinate_lift_all_levels(self):
        for region in ((0, 0, 3840, 2160), (960, 540, 2880, 1620), (480, 270, 1440, 810)):
            self.assertEqual(lift((0, 0, 960, 540), region), region)
            b = lift((240, 135, 720, 405), region)
            self.assertTrue(contains(region, b))
            self.assertAlmostEqual(iou(region, b), .25)

    def test_iou_and_size_boundaries(self):
        self.assertEqual(iou((0, 0, 10, 10), (10, 0, 20, 10)), 0)
        self.assertEqual(iou((0, 0, 0, 0), (0, 0, 0, 0)), 0)
        self.assertEqual(size_bucket((0, 0, 32, 64)), '32-64')
        self.assertFalse(contains((0, 0, 100, 100), (90, 90, 110, 110)))

    def test_gate_requires_both_recall_and_localization(self):
        rows = []
        for obj in range(30):
            for mode, level, value in [('matched', 0, .2), ('matched', 1, .2), ('native', 1, .7)]:
                rows.append(dict(frame=0, object_index=obj, mode=mode, level=level,
                                 budget=8, best_iou=value, size_source_short_side='<32'))
        self.assertEqual(gate(rows)['decision'], 'GO')
        self.assertEqual(summarize(rows)[0]['n'], 30)
        for r in rows:
            if r['mode'] == 'native':
                r['best_iou'] = .49  # better localization alone cannot pass
        self.assertEqual(gate(rows)['decision'], 'STOP')
        self.assertEqual(gate([])['decision'], 'STOP')

    def test_unpaired_objects_do_not_count(self):
        rows = [dict(frame=0, object_index=i, mode='native', level=1, budget=8, best_iou=1)
                for i in range(100)]
        self.assertEqual(gate(rows)['levels'][0]['paired_n'], 0)

    @unittest.skipUnless(platform.system() == 'Windows', 'Windows inference guard')
    def test_windows_rejects_before_output_or_ml_import(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'must-not-exist'
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('phase_a.py')),
                                     '--azure-vm', '--out', str(out)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Inference is allowed only on Azure Linux', result.stderr)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
