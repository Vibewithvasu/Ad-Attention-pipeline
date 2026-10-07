"""Pick two held-out OSIE images WITHOUT cherry-picking: one at the median v2-v1 gain, one at the 10th percentile (a failure case)."""
import json, os, sys, cv2, numpy as np, scipy.io as sio
import metrics as M, saliency_v2 as V2
from saliency_pipeline import compute_heatmap, get_face_box
OSIE = "/home/claude/osie"
recs = sio.loadmat(f"{OSIE}/data/eye/fixations.mat", squeeze_me=True, struct_as_record=False)["fixations"]
test_idx = np.random.default_rng(2026).permutation(len(recs))[:200]
net = cv2.dnn.readNetFromONNX("models/face_detector2.onnx")
sr = cv2.saliency.StaticSaliencySpectralResidual_create(); fg = cv2.saliency.StaticSaliencyFineGrained_create()
OW, OH = V2.OUT
rows = []
for i in test_idx:
    r = recs[i]; img = cv2.imread(f"{OSIE}/data/stimuli/{r.img}"); h, w = img.shape[:2]
    fix = np.zeros((OH, OW), np.float32)
    for s in r.subjects:
        for x, y in zip(np.atleast_1d(s.fix_x), np.atleast_1d(s.fix_y)):
            xi, yi = int(round(x * OW / w)), int(round(y * OH / h))
            if 0 <= xi < OW and 0 <= yi < OH: fix[yi, xi] = 1
    dens = cv2.GaussianBlur(fix, (0, 0), 6.0)
    p1 = cv2.resize(compute_heatmap(img, sr, fg, get_face_box(img, net)).astype(np.float32), V2.OUT, interpolation=cv2.INTER_AREA)
    p2 = V2.predict(V2.compute_channels(img, net))
    rows.append((int(i), r.img, M.cc(p2, dens) - M.cc(p1, dens), M.cc(p1, dens), M.cc(p2, dens), img, dens, p1, p2))
gain = np.array([r[2] for r in rows]); order = np.argsort(gain)
print("per-image CC gain v2-v1: median %.3f | p10 %.3f | share of images where v2 > v1: %.0f%%" % (np.median(gain), np.percentile(gain, 10), 100 * (gain > 0).mean()))
picks = [rows[order[len(order) // 2]], rows[order[int(0.10 * len(order))]]]
def heat(m, size):
    m = cv2.resize(m.astype(np.float32), size); m = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(m, cv2.COLORMAP_JET)
T = (420, 315); out_rows = []
for (i, name, g, c1, c2, img, dens, p1, p2) in picks:
    st = cv2.resize(img, T)
    hum = cv2.addWeighted(st, 0.4, heat(dens, T), 0.6, 0)
    v1 = cv2.addWeighted(st, 0.4, heat(p1, T), 0.6, 0)
    v2 = cv2.addWeighted(st, 0.4, heat(p2, T), 0.6, 0)
    for im, lab in ((st, f"OSIE {name}"), (hum, "Human gaze (15 viewers)"), (v1, f"v1  CC {c1:.2f}"), (v2, f"v2  CC {c2:.2f}")):
        cv2.rectangle(im, (0, 0), (T[0], 30), (0, 0, 0), -1)
        cv2.putText(im, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    out_rows.append(np.hstack([st, hum, v1, v2]))
cv2.imwrite("fig_osie.png", np.vstack(out_rows))
json.dump({"median_gain": float(np.median(gain)), "p10_gain": float(np.percentile(gain, 10)), "pct_images_v2_better": float(100 * (gain > 0).mean()),
           "picked": [(p[1], round(p[3], 3), round(p[4], 3)) for p in picks]}, open("osie_per_image.json", "w"), indent=2)
print(picks[0][1], picks[1][1])
