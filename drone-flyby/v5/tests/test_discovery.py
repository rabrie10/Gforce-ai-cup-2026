"""Exercise decoder geometry and proposal ranking with synthetic network outputs."""
import unittest
import numpy as np
from v5.discovery import Discovery


class Input:
    name='images'


class Session:
    def __init__(self,rows):
        self.rows=rows

    def run(self,outputs,feed):
        return [np.asarray(self.rows,dtype=np.float32).T[None]]


class DiscoveryContracts(unittest.TestCase):
    def detector(self,rows,budget=1):
        detector=Discovery.__new__(Discovery)
        detector.budget=budget
        detector.threshold=.005
        detector.multiscale=False
        detector.size=640
        detector.input=Input()
        detector.session=Session(rows)
        return detector

    def test_letterbox_padding_removed_before_source_conversion(self):
        # 960x540 scales to 640x360, with 140 pixels of vertical padding.
        detector=self.detector([[140.,220.,40.,40.,.9]])
        raw,selected,_=detector.propose(np.zeros((540,960,3),np.uint8),[960,540,2880,1620])
        np.testing.assert_allclose(selected[0]['local_box'],[180,90,240,150])
        np.testing.assert_allclose(selected[0]['source_box'],[1320,720,1440,840])

    def test_raw_candidates_survive_diagnostic_nms_and_budget(self):
        detector=self.detector([[140.,220.,40.,40.,.9],[141.,220.,40.,40.,.8],[300.,220.,40.,40.,.7]])
        raw,selected,metrics=detector.propose(np.zeros((540,960,3),np.uint8),[0,0,3840,2160])
        self.assertEqual(len(raw),3)
        self.assertEqual(len(selected),1)
        self.assertEqual(raw[1]['filtered_reason'],'nms')
        self.assertEqual(raw[2]['filtered_reason'],'recognition_budget')
        self.assertEqual(raw[2]['candidate_rank'],2)

    def test_general_coco_head_cannot_silently_replace_target_detector(self):
        detector=self.detector([[140.,220.,40.,40.,.9,.8]])
        with self.assertRaises(ValueError):
            detector.propose(np.zeros((540,960,3),np.uint8),[0,0,3840,2160])

    def test_degenerate_and_nan_network_boxes_are_not_emitted(self):
        detector=self.detector([[140.,220.,-1.,40.,.9],[float('nan'),220.,40.,40.,.9]])
        raw,selected,_=detector.propose(np.zeros((540,960,3),np.uint8),[0,0,3840,2160])
        self.assertEqual(selected,[])


if __name__=='__main__':
    unittest.main()
