"""Freeze manual annotations (measured on original pixels, before any detector replay).
Boxes are local observation xyxy (960x540). Global boxes derived from sidecar source_region_xyxy."""
import json, glob, hashlib, datetime, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(HERE, "..", "attempt_d09922e7a50f4cd88682f2d895fbef91", "captures")
# id, frame, level, box, family, existence_conf, difficulty, clipped, occluded, note
T = [
 ("f52_heli",52,2,[334,275,430,350],"helicopter","HIGH","EASY","",False,"Full helicopter, rotors spread, dark olive on light concrete pad. Immediately recognizable."),
 ("f55_heli",55,2,[287,475,382,540],"helicopter","HIGH","MODERATE","bottom",False,"Same helicopter type, clipped by bottom edge (~lower rotor blades cut); body+3 blades visible."),
 ("f6_armor",6,2,[213,147,262,173],"tracked_armor_like","HIGH","EASY","",False,"Olive camouflage tank-like vehicle w/ gun barrel on dry grass beside trees. Class unknown."),
 ("f7_armor",7,2,[211,213,261,239],"tracked_armor_like","HIGH","EASY","",False,"Same object as f6, next frame."),
 ("f100_armor",100,2,[753,484,811,510],"tracked_armor_like","HIGH","EASY","",False,"Dark camouflage vehicle w/ gun on sand path. Class unknown."),
 ("f204_veh_small",204,2,[112,105,155,146],"vehicle_like","HIGH","EASY","",False,"Olive camouflage vehicle on dirt yard. Class unknown."),
 ("f204_launcher_large",204,2,[320,189,422,281],"launcher_like","HIGH","EASY","",False,"Large olive multi-section launcher/SAM-truck-like object on dirt pad."),
 ("f4_armor",4,1,[829,408,853,421],"tracked_armor_like","MEDIUM","MODERATE","",False,"Small dark olive vehicle w/ barrel between trees; ~24x13 px."),
 ("f50_armor",50,1,[201,342,221,362],"tracked_armor_like","MEDIUM","MODERATE","",False,"Small olive vehicle w/ barrel in meadow, ~20x20 px."),
 ("f50_dark_b",50,1,[654,99,680,124],"vehicle_like","MEDIUM","MODERATE","",False,"Dark camouflaged object at edge of concrete pad, partly under tree shadow; class unknown."),
 ("f50_dark_a",50,1,[637,99,650,109],"vehicle_like","LOW","HARD","",False,"Tiny dark object with orange fleck on concrete pad; ~12x9 px. Uncertain."),
 ("f51_armor",51,1,[163,158,183,178],"tracked_armor_like","MEDIUM","MODERATE","",False,"Small olive vehicle w/ barrel in meadow, ~20x20 px (probably same object as f50_armor)."),
 ("f101_armor",101,1,[648,142,677,155],"tracked_armor_like","MEDIUM","MODERATE","",False,"Dark camouflage vehicle, ~29x13 px, on sand next to red-roof building (same object as f100_armor)."),
 ("f105_armor",105,1,[668,290,698,304],"tracked_armor_like","MEDIUM","MODERATE","",False,"Same as f101, ~30x14 px."),
]
UNCERTAIN = [
 {"frame_index":52,"level":2,"note":"thin lattice/pylon-like structures upper right (~x700-860,y0-170) could be 'tower'-class objects or range infrastructure; no box frozen"},
 {"frame_index":101,"level":1,"note":"rows of parked equipment at upper-left yard (~x20-260,y75-125); could be vehicles; no box frozen"},
]
NONTARGET = "cars/vans, boats on cradles (f151/f201), buildings, roofs, huts, hay bales, trees judged NOT targets; frames 102,150,151,153,159,200,201,203 have no visible plausible target"
out = {"provenance":"Manual review of original 960x540 hosted pixels (zoom crops), frozen "+datetime.datetime.now().astimezone().isoformat()+" BEFORE any ms1 replay/candidate overlay. One f204 post-merge candidate line was accidentally printed in a metadata dump after the f204 boxes were measured. Not ground truth.",
       "correction_note":"Earlier session annotation 'hosted_f55_helicopter' [335,267,435,355] matches the f52 helicopter location, not f55. The f55 helicopter is at bottom edge [287,475,382,540].",
       "targets":[], "uncertain_no_box":UNCERTAIN, "non_target_note":NONTARGET}
for tid,fr,lv,b,fam,conf,diff,clip,occ,note in T:
    f = glob.glob(os.path.join(CAP, f"*_{fr}_{lv}_*.json"))[0]
    sc = json.load(open(f)); x0,y0,x1,y1 = sc["received_camera"]["source_region_xyxy"]; s=(x1-x0)/960
    g=[x0+b[0]*s,y0+b[1]*s,x0+b[2]*s,y0+b[3]*s]
    w,h=b[2]-b[0],b[3]-b[1]
    out["targets"].append(dict(target_id=tid,frame_index=fr,level=lv,box_local_xyxy=b,box_source_xyxy=g,region_xyxy=[x0,y0,x1,y1],
        family=fam,existence_confidence=conf,difficulty=diff,px_w=w,px_h=h,short_side=min(w,h),long_side=max(w,h),
        source_px_short=min(w,h)*s,source_px_long=max(w,h)*s,edge_clipped=clip,partial_occlusion=occ,note=note))
p=os.path.join(HERE,"manual_annotations_frozen.json")
if os.path.exists(p): sys.exit("already frozen; refusing to overwrite")
json.dump(out,open(p,"w"),indent=1)
print(hashlib.sha256(open(p,'rb').read()).hexdigest(), len(out["targets"]))
