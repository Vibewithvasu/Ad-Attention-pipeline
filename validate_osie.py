"""
validate_osie.py
================
Scores the pipeline's saliency against REAL human eye-tracking (OSIE dataset,
700 images x 15 viewers, Xu et al. 2014, MIT licence) and fits the v2 fusion
weights on a TRAIN split, reporting only on a HELD-OUT TEST split.

Usage:
    git clone --depth 1 https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels osie
    python3 validate_osie.py osie

Outputs: osie_results.json, saliency_v2_weights.json
"""
import json
import os
import sys
import time
import cv2
import numpy as np
import scipy.io as sio
from scipy.optimize import nnls

import metrics as M
import saliency_v2 as V2
from saliency_pipeline import compute_heatmap, get_face_box

OSIE = sys.argv[1] if len(sys.argv) > 1 else "osie"
OUT_W, OUT_H = V2.OUT
SIGMA = 6.0                                   # ~1 degree of visual angle at 200px width
rng = np.random.default_rng(2026)


def density_and_fix(subjects, scale_x, scale_y):
    fix = np.zeros((OUT_H, OUT_W), np.float32)
    for s in subjects:
        xs = np.atleast_1d(s.fix_x) * scale_x
        ys = np.atleast_1d(s.fix_y) * scale_y
        for x, y in zip(xs, ys):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < OUT_W and 0 <= yi < OUT_H:
                fix[yi, xi] = 1
    dens = cv2.GaussianBlur(fix, (0, 0), SIGMA)
    return dens, fix


def load_records():
    mat = sio.loadmat(os.path.join(OSIE, "data/eye/fixations.mat"), squeeze_me=True, struct_as_record=False)
    return mat["fixations"]


def main():
    t0 = time.time()
    recs = load_records()
    net = cv2.dnn.readNetFromONNX("models/face_detector2.onnx")
    sr = cv2.saliency.StaticSaliencySpectralResidual_create()
    fg = cv2.saliency.StaticSaliencyFineGrained_create()

    n = len(recs)
    order = rng.permutation(n)
    test_idx = set(order[:200].tolist())          # 200 held-out images, fixed seed
    data = []
    for i, r in enumerate(recs):
        img = cv2.imread(os.path.join(OSIE, "data/stimuli", r.img))
        h, w = img.shape[:2]
        dens, fix = density_and_fix(r.subjects, OUT_W / w, OUT_H / h)
        ch = V2.compute_channels(img, net)
        # --- v1 (original pipeline) heatmaps at full res, then downsample to scoring res
        fb = get_face_box(img, net)
        v1_face = compute_heatmap(img, sr, fg, fb)
        v1_noface = compute_heatmap(img, sr, fg, None)
        rs = lambda m: cv2.resize(m.astype(np.float32), V2.OUT, interpolation=cv2.INTER_AREA)
        # half-vs-half human agreement (lower bound on human ceiling)
        subs = list(r.subjects)
        idx = rng.permutation(len(subs))
        a = [subs[j] for j in idx[:len(subs) // 2]]
        b = [subs[j] for j in idx[len(subs) // 2:]]
        da, fa = density_and_fix(a, OUT_W / w, OUT_H / h)
        db, fb2 = density_and_fix(b, OUT_W / w, OUT_H / h)
        data.append(dict(i=i, dens=dens, fix=fix, ch=ch, v1_face=rs(v1_face), v1_noface=rs(v1_noface),
                         ha=(da, fa), hb=(db, fb2), n_face=int(fb is not None)))
        if (i + 1) % 100 == 0:
            print(f"  features {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)

    train = [d for d in data if d["i"] not in test_idx]
    test = [d for d in data if d["i"] in test_idx]

    # ---------------- fit v2 weights (NNLS) on TRAIN only ----------------
    def fit(chs):
        fr = np.random.default_rng(11)
        X, y = [], []
        for d in train:
            cols = np.stack([d["ch"][k].ravel() for k in chs], 1)
            tgt = d["dens"].ravel()
            tgt = (tgt - tgt.mean()) / (tgt.std() + 1e-6)
            sel = fr.choice(cols.shape[0], 3000, replace=False)
            X.append(cols[sel]); y.append(tgt[sel])
        X = np.concatenate(X); y = np.concatenate(y)
        Xc = np.hstack([X, np.ones((X.shape[0], 1), np.float32)])   # intercept column (dropped after)
        w_all, _ = nnls(Xc.astype(np.float64), y.astype(np.float64))
        full = {k: 0.0 for k in V2.CHANNELS}
        full.update({k: float(v) for k, v in zip(chs, w_all[:-1])})
        return full

    weights = fit(V2.CHANNELS)

    # choose blur sigma on TRAIN subset (never on test)
    def mean_cc(sigma, subset):
        vals = []
        for d in subset:
            m = sum(weights[k] * d["ch"][k] for k in V2.CHANNELS)
            if sigma > 0:
                m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma)
            vals.append(M.cc(m, d["dens"]))
        return float(np.mean(vals))
    cand = [0, 1.5, 3, 4.5, 6]
    sub = train[:150]
    scores = {s: mean_cc(s, sub) for s in cand}
    best_sigma = max(scores, key=scores.get)
    wjson = {"weights": weights, "blur_sigma": best_sigma, "fit": "NNLS on 500 OSIE train images, seed 2026",
             "sigma_search_train_cc": scores}
    json.dump(wjson, open("saliency_v2_weights.json", "w"), indent=2)

    # ---------------- evaluate on TEST ----------------
    center = V2.center_prior((OUT_H, OUT_W))
    rand_rng = np.random.default_rng(1)

    def evaluate(name, fn):
        rows = [M.all_metrics(fn(d), d["dens"], d["fix"]) for d in test]
        return name, {k: np.array([r[k] for r in rows]) for k in rows[0]}

    models = [
        evaluate("random_noise", lambda d: rand_rng.random((OUT_H, OUT_W))),
        evaluate("center_prior_only", lambda d: center),
        evaluate("v1_no_face (original SR+FG)", lambda d: d["v1_noface"]),
        evaluate("v1_with_face (original pipeline)", lambda d: d["v1_face"]),
        evaluate("v2_learned_fusion", lambda d: V2.predict(d["ch"], wjson)),
    ]
    # human agreement, half vs half (both directions)
    hh = [M.all_metrics(d["ha"][0], d["hb"][0], d["hb"][1]) for d in test] + \
         [M.all_metrics(d["hb"][0], d["ha"][0], d["ha"][1]) for d in test]
    models.append(("human_half_vs_half (7v8 viewers)", {k: np.array([r[k] for r in hh]) for k in hh[0]}))

    def boot(x, B=2000):
        r = np.random.default_rng(7)
        m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(B)]
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    out = {"n_train": len(train), "n_test": len(test), "weights": weights, "blur_sigma": best_sigma, "models": {}}
    print("\nHELD-OUT TEST (n=%d images)" % len(test))
    print(f"{'model':<38}{'CC':>8}{'SIM':>8}{'KLD':>8}{'NSS':>8}{'AUC':>8}")
    for name, res in models:
        out["models"][name] = {k: {"mean": float(np.nanmean(v)), "ci95": boot(v[~np.isnan(v)])} for k, v in res.items()}
        print(f"{name:<38}" + "".join(f"{np.nanmean(res[k]):8.3f}" for k in ["CC", "SIM", "KLD", "NSS", "AUC"]))

    # paired bootstrap: v2 - v1_with_face, per metric
    v1r = dict(models)["v1_with_face (original pipeline)"]
    v2r = dict(models)["v2_learned_fusion"]
    out["paired_v2_minus_v1"] = {}
    for k in ["CC", "SIM", "KLD", "NSS", "AUC"]:
        diff = v2r[k] - v1r[k]
        out["paired_v2_minus_v1"][k] = {"mean": float(np.nanmean(diff)), "ci95": boot(diff[~np.isnan(diff)])}
    print("\nPaired v2 - v1 (95% bootstrap CI):")
    for k, v in out["paired_v2_minus_v1"].items():
        print(f"  {k}: {v['mean']:+.3f}  [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    print("\nLearned weights:", {k: round(v, 3) for k, v in weights.items()}, "blur sigma:", best_sigma)
    # ---------------- ablations (all refit on TRAIN, scored on TEST) ----------------
    ablations = {
        "v2_without_center_prior": [c for c in V2.CHANNELS if c != "center"],
        "v2_without_colour_channels": [c for c in V2.CHANNELS if c not in ("ft", "color_dog")],
        "v2_bottom_up_only (no face, no centre)": [c for c in V2.CHANNELS if c not in ("face", "center")],
    }
    out["ablations"] = {}
    print("\nABLATIONS (refit on train, scored on the same held-out test images)")
    print(f"{'variant':<42}{'CC':>8}{'SIM':>8}{'KLD':>8}{'NSS':>8}{'AUC':>8}")
    for name, chs in ablations.items():
        w = fit(chs)
        wj = {"weights": w, "blur_sigma": best_sigma}
        _, res = evaluate(name, lambda d, wj=wj: V2.predict(d["ch"], wj))
        out["ablations"][name] = {"weights": w, **{k: float(np.nanmean(v)) for k, v in res.items()}}
        print(f"{name:<42}" + "".join(f"{np.nanmean(res[k]):8.3f}" for k in ["CC", "SIM", "KLD", "NSS", "AUC"]))
        if name == "v2_without_center_prior":
            json.dump({"weights": w, "blur_sigma": best_sigma,
                       "fit": "NNLS on OSIE train, centre prior excluded (for design-driven layouts)"},
                      open("saliency_v2_weights_nocenter.json", "w"), indent=2)

    json.dump(out, open("osie_results.json", "w"), indent=2)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
