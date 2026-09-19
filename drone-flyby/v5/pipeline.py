"""Causal full-frame V5 pipeline with independently switchable camera and state."""
from collections import OrderedDict
from dataclasses import dataclass, field
import logging
import os
import threading
import time

import numpy as np

from dtos import OBJECT_CLASSES, DroneFlybyPredictResponseDto, DroneFlybyPredictionDto
from utils import decode_view
from v2.geometry import apply_affine_to_box, box_center, contains, iou
from v2.gmc import GlobalMotionEstimator
from v2.scheduler import CameraScheduler
from v5.common import verify_assets
from v5.discovery import Discovery
from v5.visual_expert import VisualExpert


@dataclass
class Config:
    assets: str = field(default_factory=lambda:os.getenv('V5_ASSETS','/assets'))
    device: str = field(default_factory=lambda:os.getenv('V5_DEVICE','cpu'))
    threads: int = field(default_factory=lambda:int(os.getenv('V5_THREADS','2')))
    candidate_budget: int = field(default_factory=lambda:int(os.getenv('V5_CANDIDATES','32')))
    head_name: str = field(default_factory=lambda:os.getenv('V5_HEAD','visual_head.npz'))
    active_camera: bool = field(default_factory=lambda:os.getenv('V5_ACTIVE_CAMERA','0')=='1')
    temporal: bool = field(default_factory=lambda:os.getenv('V5_TEMPORAL','1')=='1')
    multiscale: bool = field(default_factory=lambda:os.getenv('V5_MULTISCALE','0')=='1')
    minimum_target: float = .5
    minimum_discovery: float = .01
    max_tracks: int = 128
    ttl: int = 3


@dataclass
class Track:
    id: int
    box: list
    posterior: np.ndarray
    feature: np.ndarray
    score: float
    first_seen: int
    last_seen: int
    evidence_level: int
    evidence_count: int = 1
    last_zoom: int = -20


class State:
    def __init__(self):
        self.gmc=GlobalMotionEstimator()
        self.tracks=[]
        self.next_id=0
        self.last_index=-1
        self.last_l0=-1
        self.zoom_dwell=0
        self.cache=OrderedDict()


class Pipeline:
    def __init__(self, config=None, discovery=None, expert=None):
        self.config=config or Config()
        if discovery is None or expert is None:
            self.manifest=verify_assets(self.config.assets)
        else:
            self.manifest={'injected_components':True}
        self.discovery=discovery or Discovery(self.config.assets,budget=self.config.candidate_budget,
            multiscale=self.config.multiscale,device=self.config.device,threads=self.config.threads)
        self.expert=expert or VisualExpert(self.config.assets,device=self.config.device,head_name=self.config.head_name,threads=self.config.threads)
        self.states=OrderedDict()
        self.lock=threading.RLock()
        self.last_diagnostics={}
        self.camera=CameraScheduler()

    def empty(self,request):
        return DroneFlybyPredictResponseDto(request_id=request.request_id,frame=request.frame,annotations=[])

    def predict(self,request):
        with self.lock:
            start=time.perf_counter()
            try:
                return self._predict(request,start)
            except Exception as exc:
                logging.exception('V5 frame failed')
                # Discard partially updated state after failure; never carry it forward.
                self.states.pop(request.sequence_id,None)
                self.last_diagnostics={'error':f'{type(exc).__name__}: {exc}','total_ms':(time.perf_counter()-start)*1000}
                return self.empty(request)

    def _predict(self,r,start):
        if r.original_width!=3840 or r.original_height!=2160:
            raise ValueError('Inherited GMC requires official 3840x2160 source dimensions')
        if self.config.candidate_budget<1 or self.config.candidate_budget>500:
            raise ValueError('Candidate budget must be in 1..500')
        state=self.states.setdefault(r.sequence_id,State())
        self.states.move_to_end(r.sequence_id)
        while len(self.states)>4:
            self.states.popitem(last=False)
        key=(r.request_id,r.frame,r.frame_index)
        if key in state.cache:
            response,diagnostics=state.cache[key]
            self.last_diagnostics={**diagnostics,'cached':True}
            return response.model_copy(deep=True)
        if r.frame_index<=state.last_index:
            self.last_diagnostics={'reason':'out_of_order','frame':r.frame,'total_ms':(time.perf_counter()-start)*1000}
            return self.empty(r)
        image=decode_view(r.view)
        if image.shape[:2]!=(540,960):
            raise ValueError('Received image must be 960x540')
        decoded=time.perf_counter()
        region=r.view.source_region_xyxy
        if not(0<=region[0]<region[2]<=3840 and 0<=region[1]<region[3]<=2160):
            raise ValueError('Invalid source region')
        if not self.config.temporal:
            state.tracks.clear()
        motion=state.gmc.estimate(image,region)
        gap=r.frame_index-state.last_index if state.last_index>=0 else 1
        state.tracks=[t for t in state.tracks if r.frame_index-t.last_seen<=self.config.ttl]
        if motion.ok:
            for t in state.tracks:
                t.box=list(apply_affine_to_box(motion.matrix,t.box))
        raw,selected,stages=self.discovery.propose(image,region)
        recognition,features,visual_times=self.expert.classify(image,[p['local_box'] for p in selected])
        stages.update(visual_times)
        used=set()
        observed=set()
        associations=[]
        for p,result,feature in zip(selected,recognition,features):
            p.update(result)
            p['emitted']=False
            if result['target_probability']<self.config.minimum_target:
                p['filtered_reason']='visual_background'
                continue
            if p['discovery_score']<self.config.minimum_discovery:
                p['filtered_reason']='discovery_threshold'
                continue
            posterior=np.array(result['class_scores'],dtype=np.float32)
            score=float(p['discovery_score']*result['target_probability'])
            matches=[(iou(p['source_box'],t.box),t) for t in state.tracks if t.id not in used]
            overlap,track=max(matches,key=lambda v:v[0],default=(0.,None))
            if overlap<.2:
                if len(state.tracks)>=self.config.max_tracks:
                    p['filtered_reason']='track_capacity'
                    continue
                track=Track(state.next_id,list(p['source_box']),posterior,feature.copy(),score,
                    r.frame_index,r.frame_index,r.view.resolution_level)
                state.next_id+=1
                state.tracks.append(track)
                novel=True
            else:
                # Near-identical embeddings at the same/lower resolution carry no new
                # identity evidence. EMA bounds certainty and allows later reversal.
                novel=float(np.dot(track.feature,feature))<.995 or r.view.resolution_level>track.evidence_level
                if novel:
                    track.posterior=.45*track.posterior+.55*posterior
                    track.feature=feature.copy()
                    track.evidence_level=r.view.resolution_level
                    track.evidence_count+=1
                track.box=list(p['source_box'])
                track.last_seen=r.frame_index
                track.score=score
            p['track_id']=track.id
            p['novel_evidence']=novel
            used.add(track.id)
            observed.add(track.id)
            associations.append((p,track))
        annotations=[]
        emitted=[]
        ranked=sorted(state.tracks,key=lambda t:t.score*float(t.posterior.max()),reverse=True)
        for t in ranked:
            age=r.frame_index-t.last_seen
            # Missing inside-view objects are not emitted from stale state. Off-crop
            # retention requires a successful causal global transform.
            if t.id not in observed and (not motion.ok or intersects_region(t.box,region)):
                continue
            box=np.clip(np.array(t.box)/np.array([r.original_width,r.original_height,r.original_width,r.original_height]),0,1)
            if not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:
                continue
            cls=int(t.posterior.argmax())
            confidence=float(np.clip(t.score*t.posterior[cls]*(.8**age),0,1))
            if any(cls==c and iou(box,b)>.5 for c,b,_ in emitted):
                continue
            emitted.append((cls,box,t.id))
            annotations.append(DroneFlybyPredictionDto(object_id=OBJECT_CLASSES[cls],bbox=box.tolist(),confidence=confidence))
            if len(annotations)==500:
                break
        emitted_ids={t for _,_,t in emitted}
        for p,t in associations:
            p['emitted']=t.id in emitted_ids
            p['filtered_reason']=None if p['emitted'] else 'output_suppression'
        camera=self._camera(r,state,motion.ok) if self.config.active_camera else None
        response=DroneFlybyPredictResponseDto(request_id=r.request_id,frame=r.frame,
            annotations=annotations,requested_view=camera)
        stages.update(decode_ms=(decoded-start)*1000,gmc_ms=motion.elapsed_ms,total_ms=(time.perf_counter()-start)*1000)
        self.last_diagnostics={'frame':r.frame,'frame_index':r.frame_index,'sequence_id':r.sequence_id,
            'source_region':region,'candidates':raw,'timing':stages,'motion':motion.summary(),'frame_gap':gap,
            'tracks':[{'id':t.id,'box':t.box,'class_scores':t.posterior.tolist(),'last_seen':t.last_seen,
                       'evidence_count':t.evidence_count} for t in state.tracks],
            'response':response.model_dump()}
        state.last_index=r.frame_index
        state.cache[key]=(response.model_copy(deep=True),self.last_diagnostics)
        while len(state.cache)>8:
            state.cache.popitem(last=False)
        return response

    def _camera(self,r,state,motion_ok):
        level=r.view.resolution_level
        if level==0:
            state.last_l0=r.frame_index
            state.zoom_dwell=0
        else:
            state.zoom_dwell+=1
        allowed=r.camera_constraints.allowed_resolution_levels
        current=(r.view.center_x,r.view.center_y)
        # Account for skipped frames in coverage age. L2 legally returns via L1.
        if not motion_ok or r.frame_index-state.last_l0>=4 or state.zoom_dwell>=3:
            next_level=0 if 0 in allowed else 1
            return self.camera._plan_move(next_level,(1920,1080),current,r.camera_constraints)
        candidates=[]
        for t in state.tracks:
            if r.frame_index-t.last_seen>1 or r.frame_index-t.first_seen<1 or (level==0 and r.frame_index-t.last_zoom<6):
                continue
            entropy=float(-(t.posterior*np.log(np.maximum(t.posterior,1e-12))).sum()/np.log(16))
            if entropy<.35:
                continue
            next_level=min(2,level+1)
            if next_level not in allowed:
                continue
            coverage_penalty=sum(.05 for other in state.tracks if other.id!=t.id and not contains(region_box(next_level,box_center(t.box)),other.box))
            utility=entropy+min(1.,32/max(1.,min(t.box[2]-t.box[0],t.box[3]-t.box[1])))-coverage_penalty-.15*next_level
            candidates.append((utility,t,next_level))
        if not candidates:
            if level>0:
                return self.camera._plan_move(0 if 0 in allowed else 1,(1920,1080),current,r.camera_constraints)
            return None
        _,target,next_level=max(candidates,key=lambda q:q[0])
        command=self.camera._plan_move(next_level,box_center(target.box),current,r.camera_constraints)
        if command is not None:
            target.last_zoom=r.frame_index
        return command


def intersects_region(a,b):
    return min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1])


def region_box(level,center):
    w,h=(3840//(2**level),2160//(2**level))
    return (center[0]-w/2,center[1]-h/2,center[0]+w/2,center[1]+h/2)
