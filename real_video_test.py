"""
real_video_test.py -- v1 vs v2 motion on REAL open footage.

Clips (all open / sample material, used for research & education):
  vtest.avi        OpenCV sample (static camera, walking pedestrians)
  bigbuckbunny.mp4 (c) Blender Foundation, CC-BY 3.0  (via scikit-video)
  bikes.mp4        scikit-video sample (moving camera)

Proxy ground truth for the static-camera clip = MOG2 background subtraction
(Zivkovic 2004). That is an independent MOVEMENT signal, NOT human attention.
"""
import json
import os
import sys
import cv2
import numpy as np
import skvideo
import motion_emd as V1
import motion_v2 as V2

D = os.path.join(os.path.dirname(skvideo.__file__), "datasets", "data")
CLIPS = {"vtest": "/tmp/vtest.avi",
         "bigbuckbunny": os.path.join(D, "bigbuckbunny.mp4"),
         "bikes": os.path.join(D, "bikes.mp4")}
TW = 640
THR2 = V2.calibrate_threshold()


def read(path, max_frames=400):
    cap = cv2.VideoCapture(path)
    out = []
    while len(out) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        out.append(cv2.resize(f, (TW, int(h * TW / w))))
    cap.release()
    return out


def auc_rank(score, mask):
    """ROC-AUC of score separating mask pixels from non-mask (Mann-Whitney)."""
    s = score.ravel().astype(np.float64)
    m = mask.ravel().astype(bool)
    n1, n0 = int(m.sum()), int((~m).sum())
    if n1 < 50 or n0 < 50:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = csum - (cnt - 1) / 2.0
    ranks = avg[inv]
    return float((ranks[m].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    res = {}
    figs = {}
    for name, path in CLIPS.items():
        frames = read(path)
        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        n = len(frames)
        r = {"frames": n, "size": f"{frames[0].shape[1]}x{frames[0].shape[0]}"}
        flood1 = flood2 = cam = 0
        act1, act2 = [], []
        aucs1, aucs2, bgleak2 = [], [], []
        mog = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=True)
        pick = {}
        for t in range(n - 1):
            fg = mog.apply(frames[t + 1])
            e1, _ = V1.reichardt_motion_map(gray[t], gray[t + 1])
            e1 = e1.astype(np.float32)
            rr = V2.motion_saliency(gray[t], gray[t + 1])
            e2 = rr["energy"]
            thr2 = V2.frame_threshold(e2, THR2)
            a1 = 100 * float((e1 > 15).mean()); a2 = 100 * float((e2 > thr2).mean())
            act1.append(a1); act2.append(a2)
            flood1 += a1 > 20; flood2 += a2 > 20; cam += rr["ego"] is not None
            if name == "vtest" and t > 40:
                m = (fg > 200).astype(np.uint8)
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                m = cv2.dilate(m, np.ones((7, 7), np.uint8))
                x1, x2 = auc_rank(e1, m), auc_rank(e2, m)
                if x1 is not None and x2 is not None:
                    aucs1.append(x1); aucs2.append(x2)
                    sup = e2 > thr2
                    tot = float(e2[sup].sum()) + 1e-9
                    mm = cv2.dilate(m, np.ones((25, 25), np.uint8)).astype(bool)
                    bgleak2.append(float(e2[sup & ~mm].sum()) / tot)
            if t in (int(n * 0.3), int(n * 0.6)):
                pick[t] = (frames[t + 1].copy(), e1, e2)
        r["pct_pixels_flagged_mean"] = {"v1": round(float(np.mean(act1)), 1), "v2": round(float(np.mean(act2)), 1)}
        r["frames_with_>20pct_flagged_(pan_flooding)"] = {"v1": f"{flood1}/{n-1}", "v2": f"{flood2}/{n-1}"}
        r["camera_motion_compensated_frames"] = f"{cam}/{n-1}"
        if aucs1:
            r["MOG2_proxy_AUC"] = {"v1": round(float(np.mean(aucs1)), 3), "v2": round(float(np.mean(aucs2)), 3), "n_frames": len(aucs1)}
            d = np.array(aucs2) - np.array(aucs1)
            rng = np.random.default_rng(3)
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            r["MOG2_proxy_AUC_gain_v2_minus_v1_ci95"] = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
            r["v2_energy_outside_foreground_pct"] = round(100 * float(np.mean(bgleak2)), 1)
        res[name] = r
        print(name, json.dumps(r, indent=1))
        figs[name] = pick

    # figure panels: frame | v1 | v2 (fixed scales)
    for name, pick in figs.items():
        rows = []
        for t, (fr, e1, e2) in sorted(pick.items())[:1]:
            h, w = fr.shape[:2]
            v1 = cv2.applyColorMap(np.clip(e1, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            v2 = cv2.applyColorMap(np.clip(255 * e2 / 3.0, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            for img, lab in ((fr, "Frame"), (v1, "v1 motion"), (v2, "v2 motion")):
                cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
                cv2.putText(img, lab, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            rows.append(np.hstack([fr, v1, v2]))
        cv2.imwrite(f"real_{name}.png", rows[0])
    json.dump(res, open("real_video_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
