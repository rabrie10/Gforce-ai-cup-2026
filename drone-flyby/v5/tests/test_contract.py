"""Model-free tests of coordinate, causal, safety and camera contracts."""
import unittest
import numpy as np
from dtos import DroneFlybyPredictRequestDto,OBJECT_CLASSES
from local_evaluator import Camera,build_request
from utils import encode_image
from v2.geometry import view_to_source
from v2.gmc import MotionEstimate
from v5.pipeline import Pipeline,Config


class FakeDiscovery:
    def __init__(self):
        self.boxes=[[200.,100.,220.,120.]]
        self.calls=0

    def propose(self,image,region):
        self.calls+=1
        result=[{'local_box':b,'source_box':list(view_to_source(b,region)),'discovery_score':.9,
            'selected':True,'candidate_rank':i+1} for i,b in enumerate(self.boxes)]
        return result,result,{}


class FakeExpert:
    def __init__(self):
        self.cls=0
        self.target=.95

    def classify(self,image,boxes):
        p=np.ones(16)*.01
        p[self.cls]=.85
        p/=p.sum()
        feature=np.zeros(384,dtype=np.float32)
        feature[self.cls]=1
        results=[{'class_scores':p.tolist(),'target_probability':self.target,
                  'top3':[{'class':OBJECT_CLASSES[self.cls],'score':float(p.max())}]} for _ in boxes]
        return results,np.stack([feature for _ in boxes]) if boxes else np.empty((0,384)),{}


class FakeMotion:
    def __init__(self,dx=0,dy=0,ok=True):
        self.dx,self.dy,self.ok=dx,dy,ok

    def estimate(self,image,region):
        return MotionEstimate(matrix=np.array([[1.,0.,self.dx],[0.,1.,self.dy]]),ok=self.ok)


def request(index=0,camera=None,sequence='test'):
    camera=camera or Camera()
    payload=build_request(index,index,camera,encode_image(np.zeros((540,960,3),dtype=np.uint8)),None)
    payload['sequence_id']=sequence
    return DroneFlybyPredictRequestDto.model_validate(payload)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.discovery=FakeDiscovery()
        self.expert=FakeExpert()
        self.pipeline=Pipeline(Config(),self.discovery,self.expert)

    def test_l1_global_geometry(self):
        cam=Camera(1,1920,1080)
        r=request(camera=cam)
        response=self.pipeline.predict(r)
        self.assertEqual(response.request_id,r.request_id)
        self.assertEqual(response.frame,r.frame)
        expected=np.array([1360,740,1400,780])/np.array([3840,2160,3840,2160])
        np.testing.assert_allclose(response.annotations[0].bbox,expected)

    def test_duplicate_request_does_not_accumulate_evidence(self):
        r=request()
        a=self.pipeline.predict(r)
        b=self.pipeline.predict(r)
        self.assertEqual(a,b)
        self.assertEqual(self.discovery.calls,1)

    def test_repeated_appearance_does_not_accumulate_identity(self):
        self.pipeline.predict(request())
        self.pipeline.states['test'].gmc=FakeMotion()
        for i in range(1,5):
            self.pipeline.predict(request(i))
        self.assertEqual(self.pipeline.states['test'].tracks[0].evidence_count,1)

    def test_better_later_identity_can_replace_early_class(self):
        self.pipeline.predict(request())
        self.pipeline.states['test'].gmc=FakeMotion()
        self.expert.cls=2
        response=self.pipeline.predict(request(1))
        self.assertEqual(response.annotations[0].object_id,OBJECT_CLASSES[2])
        self.assertGreater(self.pipeline.states['test'].tracks[0].posterior[0],0)

    def test_rewind_cannot_expose_future_tracks(self):
        self.pipeline.predict(request(10))
        response=self.pipeline.predict(request(2))
        self.assertEqual(response.annotations,[])

    def test_skipped_frames_expire_stale_tracks(self):
        self.pipeline.predict(request())
        self.pipeline.states['test'].gmc=FakeMotion()
        self.discovery.boxes=[]
        self.assertEqual(self.pipeline.predict(request(9)).annotations,[])
        self.assertEqual(self.pipeline.states['test'].tracks,[])

    def test_off_crop_track_requires_valid_motion(self):
        self.pipeline.predict(request())
        self.pipeline.states['test'].gmc=FakeMotion(dx=20)
        self.discovery.boxes=[]
        response=self.pipeline.predict(request(1,Camera(2,2880,1350)))
        self.assertEqual(len(response.annotations),1)
        self.assertAlmostEqual(response.annotations[0].bbox[0],820/3840)
        self.pipeline.states['test'].gmc=FakeMotion(ok=False)
        self.assertEqual(self.pipeline.predict(request(2,Camera(2,2880,1350))).annotations,[])

    def test_background_is_rejected_without_unknown_class(self):
        self.expert.target=.1
        self.assertEqual(self.pipeline.predict(request()).annotations,[])

    def test_same_class_duplicate_suppression(self):
        self.discovery.boxes*=2
        response=self.pipeline.predict(request())
        self.assertEqual(len(response.annotations),1)

    def test_corrupt_image_empty_echo_and_state_reset(self):
        r=request()
        r.view.image='not_an_image'
        response=self.pipeline.predict(r)
        self.assertEqual(response.request_id,r.request_id)
        self.assertEqual(response.frame,r.frame)
        self.assertEqual(response.annotations,[])
        self.assertNotIn('test',self.pipeline.states)

    def test_sequences_are_isolated_and_bounded(self):
        for i in range(10):
            response=self.pipeline.predict(request(sequence=f'seq{i}'))
            self.assertEqual(len(response.annotations),1)
        self.assertEqual(len(self.pipeline.states),4)

    def test_camera_returns_legally_from_l2(self):
        self.pipeline.config.active_camera=True
        cam=Camera(2,3360,1890)
        r=request(camera=cam)
        response=self.pipeline.predict(r)
        self.assertIsNotNone(response.requested_view)
        command=response.requested_view
        cam.apply(command.resolution_level,command.center_x,command.center_y)
        self.assertEqual(command.resolution_level,1)


if __name__=='__main__':
    unittest.main()
