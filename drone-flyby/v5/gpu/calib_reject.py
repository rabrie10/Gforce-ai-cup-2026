import os, json, cv2, numpy as np
from v5.pipeline import Config
from v5.gpu.discovery_v54 import MergedDiscovery, RejectingExpert

cfg = Config()
disc = MergedDiscovery(cfg.assets, budget=cfg.candidate_budget, device=cfg.device)
expert = RejectingExpert(cfg.assets, device=cfg.device, head_name=cfg.head_name)
files = {60: "00060_80324c03bcc65ff1d04c510e_180324c03bcc65ff1d04c510e_60_0_1920_1080.png",
         125: "00125_80324c03bcc65ff1d04c510e_80324c03bcc65ff1d04c510e_125_0_1920_1080.png"}
manual = json.load(open("/workspace/results/manual_targets.json"))["targets"]
mby = {}
for t in manual:
    mby.setdefault(t["capture"], []).append(t)

# gather selected detector crops (proposals) and oracle target crops
prop_feats = []
for cap, fn in files.items():
    img = cv2.imread(f"/workspace/hosted-v3/images/{fn}")
    raw, sel, _ = disc.propose(img, [0, 0, 3840, 2160])
    _, feats, _ = expert.classify(img, [c["local_box"] for c in sel])
    prop_feats.append(feats)
prop_feats = np.concatenate(prop_feats)
# oracle target crops (approx manual boxes)
orc_feats = []
for cap, fn in files.items():
    img = cv2.imread(f"/workspace/hosted-v3/images/{fn}")
    _, feats, _ = expert.classify(img, [t["box"] for t in mby[cap]])
    orc_feats.append(feats)
orc_feats = np.concatenate(orc_feats)

base_prop = expert.classify_features(prop_feats)  # tau=0 baseline
base_orc = expert.classify_features(orc_feats)
print("proposal crops:", len(prop_feats), " oracle target crops:", len(orc_feats))
print("tau | proposals_emitted(target>0.5) | oracle_targets_retained(target>0.5)")
for tau in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]:
    expert.ref_tau = tau
    p = expert.classify_features(prop_feats)
    o = expert.classify_features(orc_feats)
    pe = sum(1 for r in p if r["target_probability"] >= 0.5)
    oe = sum(1 for r in o if r["target_probability"] >= 0.5)
    print(f"{tau:.2f} | {pe}/{len(p)} | {oe}/{len(o)}")
# show ref_max distribution
rm_prop = sorted(max(r["reference_similarity"]) for r in base_prop)
rm_orc = sorted(max(r["reference_similarity"]) for r in base_orc)
print("proposal ref_max pctiles 10/50/90:", [round(np.percentile(rm_prop, q), 3) for q in (10, 50, 90)])
print("oracle   ref_max pctiles 10/50/90:", [round(np.percentile(rm_orc, q), 3) for q in (10, 50, 90)])
