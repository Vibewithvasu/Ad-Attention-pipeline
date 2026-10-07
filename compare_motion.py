"""
compare_motion.py -- identical probes, v1 (motion_emd) vs v2 (motion_v2).
All clips are synthetic with known ground truth.
"""
import json
import cv2
import numpy as np
import game_motion_test as g
import motion_emd as V1
import motion_v2 as V2

H, W, N = g.H, g.W, g.N
THR2 = V2.calibrate_threshold()
THR1 = 15  # as shipped in motion_emd.detect_motion_region
print(f"v2 noise-calibrated threshold = {THR2:.3f}")


# ---------- detector wrappers -> (energy_map, regions[list of box]) ----------
def det_v1(f0, f1):
    box, e, _ = V1.detect_motion_region(f0, f1, THR1)
    return e.astype(np.float32), ([box] if box else []), THR1


def det_v2(f0, f1):
    r = V2.motion_saliency(f0, f1)
    thr = V2.frame_threshold(r["energy"], THR2)
    regs = V2.detect_regions(r["energy"], thr, top_k=5)
    return r["energy"], [x["box"] for x in regs], thr


DETS = {"v1": det_v1, "v2": det_v2}


def center_in(box, gt, margin=20):
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    return gt[0] - margin <= cx <= gt[0] + gt[2] + margin and gt[1] - margin <= cy <= gt[1] + gt[3] + margin


def box_energy(e, thr, box):
    x, y, w, h = box
    sub = e[max(y, 0):y + h, max(x, 0):x + w]
    return float(sub[sub > thr].sum())


def run_clip(frames, gts, det):
    """frames: list; gts: list (per frame) of list of GT boxes (may be empty)."""
    act, share, hit, nreg = [], [], [], []
    recall = []
    for t in range(len(frames) - 1):
        e, regs, thr = det(frames[t], frames[t + 1])
        sup = e > thr
        act.append(100 * float(sup.mean()))
        tot = float(e[sup].sum()) + 1e-9
        gt = gts[t]
        if gt:
            share.append(sum(box_energy(e, thr, b) for b in gt) / tot)
            hit.append(float(bool(regs) and center_in(regs[0], gt[0]) if len(gt) == 1 else bool(regs) and any(center_in(regs[0], b) for b in gt)))
            recall.append(sum(any(center_in(r, b) for r in regs) for b in gt) / len(gt))
        nreg.append(len(regs))
    out = {"active_pct": round(float(np.mean(act)), 1), "regions_per_frame": round(float(np.mean(nreg)), 2)}
    if share:
        out["target_share_pct"] = round(100 * float(np.mean(share)), 1)
        out["top_region_on_target_pct"] = round(100 * float(np.mean(hit)), 1)
        out["mover_recall_pct"] = round(100 * float(np.mean(recall)), 1)
    return out


# ---------- probe clips ----------
def clip_pan_only():
    fr = [g.crop(g.WORLD, 200 + t * 6) for t in range(N)]
    return fr, [[] for _ in fr]


def clip_pan_fighters():
    fr, gts = [], []
    for t in range(N):
        f = g.crop(g.WORLD, 200 + t * 6)
        cv2.circle(f, (250 + 8 * (t % 5), 200), 22, 250, -1)
        cv2.circle(f, (390 - 8 * (t % 5), 200), 22, 250, -1)
        fr.append(f); gts.append([(220, 170, 200, 60)])
    return fr, gts


def clip_slow():
    fr, gts = [], []
    for t in range(N):
        f = g.crop(g.WORLD, 200 + t * 1.0)
        cx = 320 + 40 * np.sin(t / 10)
        cv2.circle(f, (int(cx), 200), 16, 250, -1)
        fr.append(f); gts.append([(int(cx) - 30, 170, 60, 60)])
    return fr, gts


def clip_noisy(sigma=4.0):
    fr, gts = clip_slow()
    r = np.random.default_rng(5)
    fr = [np.clip(f + r.normal(0, sigma, f.shape), 0, 255).astype(np.uint8) for f in fr]
    return fr, gts


def clip_zoom():
    fr, gts = [], []
    big = cv2.resize(g.WORLD[:, 400:400 + W * 2], (W * 2, H * 2)).astype(np.uint8)
    for t in range(N):
        s = 1.012 ** t
        M = cv2.getRotationMatrix2D((W, H), 0, s)
        f = cv2.warpAffine(big, M, (W * 2, H * 2), borderMode=cv2.BORDER_REPLICATE)[H // 2:H // 2 + H, W // 2:W // 2 + W].copy()
        cx = 120 + t * 4
        cv2.circle(f, (cx, 100), 18, 250, -1)
        fr.append(f); gts.append([(cx - 30, 70, 60, 60)])
    return fr, gts


def clip_speed(v):
    fr, gts = [], []
    for t in range(N):
        f = g.crop(g.WORLD, 300)
        cx = 40 + (t * v) % (W - 80)
        cv2.circle(f, (int(cx), 180), 18, 250, -1)
        fr.append(f); gts.append([(int(cx) - 35, 145, 70, 70)])
    return fr, gts


def clip_multi():
    fr, gts = [], []
    for t in range(N):
        f = g.crop(g.WORLD, 300)
        a = (80 + t * 3, 80); b = (520 - t * 4, 180); c = (300, 40 + t * 4)
        for (x, y) in (a, b, c):
            cv2.circle(f, (int(x), int(y)), 16, 250, -1)
        fr.append(f); gts.append([(int(x) - 30, int(y) - 30, 60, 60) for (x, y) in (a, b, c)])
    return fr, gts


def lone_bar(kind, length=300):
    flat = np.full((H, W), 60, np.uint8)
    fr = []
    for t in range(N):
        f = flat.copy()
        if kind == "hbar_vertical":
            y = 30 + t * 4
            cv2.rectangle(f, (170, y), (170 + length, y + 6), 250, -1)
        else:
            x = 30 + t * 4
            cv2.rectangle(f, (x, 30), (x + 6, 30 + length), 250, -1)
        fr.append(f)
    return fr


def lone_disc(bg, fg):
    fr = []
    for t in range(N):
        f = np.full((H, W), bg, np.uint8)
        cv2.circle(f, (100 + t * 4, 180), 20, fg, -1)
        fr.append(f)
    return fr


def raw_energy_sum(frames, version):
    s = 0.0
    for t in range(len(frames) - 1):
        if version == "v1":
            _, d = V1.reichardt_motion_map(frames[t], frames[t + 1])
            s += float(np.abs(d).sum())
        else:
            s += float(V2.motion_saliency(frames[t], frames[t + 1])["energy"].sum())
    return s


if __name__ == "__main__":
    res = {"threshold_v2": THR2, "probes": {}}
    probes = {
        "pan_only (no object)": clip_pan_only,
        "fast pan + 2 bright fighters": clip_pan_fighters,
        "slow explore": clip_slow,
        "slow explore + sensor noise": clip_noisy,
        "camera zoom + mover": clip_zoom,
        "3 simultaneous movers": clip_multi,
    }
    for name, fn in probes.items():
        frames, gts = fn()
        res["probes"][name] = {v: run_clip(frames, gts, d) for v, d in DETS.items()}
        print(name, json.dumps(res["probes"][name]))

    # speed tuning
    res["speed_tuning_top_region_on_target_pct"] = {}
    for v in (1, 2, 4, 8, 16):
        frames, gts = clip_speed(v)
        res["speed_tuning_top_region_on_target_pct"][f"{v}px/frame"] = {
            k: run_clip(frames, gts, d).get("top_region_on_target_pct") for k, d in DETS.items()}
        print("speed", v, res["speed_tuning_top_region_on_target_pct"][f"{v}px/frame"])

    # direction + brightness (raw energy ratios)
    out = {}
    for ver in ("v1", "v2"):
        eh = raw_energy_sum(lone_bar("hbar_vertical"), ver)
        ev = raw_energy_sum(lone_bar("vbar_horizontal"), ver)
        out[ver] = {"vertical_bar_moving_right_over_horizontal_bar_moving_down": round(ev / max(eh, 1e-9), 2)}
        dark = raw_energy_sum(lone_disc(20, 120), ver)
        bright = raw_energy_sum(lone_disc(130, 230), ver)
        out[ver]["equal_contrast_bright_over_dark"] = round(bright / max(dark, 1e-9), 2)
    res["ratios (1.0 = ideal)"] = out
    print(json.dumps(out, indent=1))
    json.dump(res, open("motion_v2_results.json", "w"), indent=2)
