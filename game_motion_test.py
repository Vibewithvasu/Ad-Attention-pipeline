"""
Stress test of motion_emd.py on game-like motion profiles.
All clips are SYNTHETIC with known ground truth (not real gameplay footage).
"""
import json
import cv2
import numpy as np
from motion_emd import reichardt_motion_map, detect_motion_region

H, W, N = 360, 640, 60
rng = np.random.default_rng(7)


def textured_world(h=H, w=W * 3):
    n = rng.random((h, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 3) * 255
    n = cv2.normalize(n, None, 40, 200, cv2.NORM_MINMAX)
    return n


WORLD = textured_world()


def crop(world, x0):
    return world[:, int(x0):int(x0) + W].copy().astype(np.uint8)


def draw_hud(f):
    cv2.rectangle(f, (20, 20), (220, 44), 255, -1)           # health bar
    cv2.putText(f, "SCORE 004210", (440, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    cv2.circle(f, (70, 300), 38, 255, 3)                      # minimap ring


def blob(f, cx, cy, r, val):
    cv2.circle(f, (int(cx), int(cy)), r, val, -1)


def make_clip(name):
    frames, boxes = [], []
    for t in range(N):
        if name == "slow_explore":
            f = crop(WORLD, 200 + t * 1.0)                   # gentle camera pan
            cx = 320 + 40 * np.sin(t / 10)
            blob(f, cx, 200, 16, 250)                        # small character
            box = (int(cx) - 30, 170, 60, 60)
        elif name == "fast_combat":
            shake = rng.integers(-6, 7)
            f = crop(WORLD, 200 + t * 6 + shake)             # fast pan + shake
            blob(f, 250 + 8 * (t % 5), 200, 22, 250)         # fighter A
            blob(f, 390 - 8 * (t % 5), 200, 22, 250)         # fighter B
            box = (210, 170, 220, 60)
        elif name == "hud_pulse":
            f = crop(WORLD, 300)                             # static camera
            draw_hud(f)
            r = 14 + int(8 * abs(np.sin(t / 4)))
            cv2.circle(f, (520, 260), r, 255, -1)            # pulsing CTA-like dot
            cx = 320 + t * 3
            blob(f, cx, 160, 14, 250)                        # slow mover
            box = (int(cx) - 25, 130, 50, 60)
        elif name == "pure_pan_no_object":
            f = crop(WORLD, 200 + t * 4)                     # camera moves, no object
            draw_hud(f)
            box = None
        elif name == "horizontal_vs_vertical":
            f = crop(WORLD, 300)
            blob(f, 160 + t * 4, 120, 18, 250)               # horizontal mover
            blob(f, 480, 60 + t * 4, 18, 250)                # vertical mover, same speed
            box = None
        elif name == "dim_vs_bright":
            f = crop(WORLD, 300)
            blob(f, 160 + t * 4, 120, 18, 250)               # bright mover
            blob(f, 160 + t * 4, 260, 18, 90)                # dim mover, same speed
            box = None
        frames.append(f.astype(np.uint8))
        boxes.append(box)
    return frames, boxes


def region_energy(energy, box):
    x, y, w, h = box
    return float(energy[max(y, 0):y + h, max(x, 0):x + w].sum())


def run(name):
    frames, boxes = make_clip(name)
    hits, share, hud_e, tot_e = 0, [], [], []
    area_frac = []
    for t in range(N - 1):
        box_pred, energy, direction = detect_motion_region(frames[t], frames[t + 1])
        tot = float(energy.sum()) + 1e-9
        tot_e.append(tot)
        # HUD zone energy (top bar + minimap) -- should be ~0 since HUD is static
        hud_e.append(float(energy[20:44, 20:220].sum() + energy[262:338, 32:108].sum()))
        area_frac.append(float((energy > 15).mean()))
        gt = boxes[t]
        if gt is not None:
            share.append(region_energy(energy, gt) / tot)
            if box_pred is not None:
                px, py, pw, ph = box_pred
                cx, cy = px + pw / 2, py + ph / 2
                if gt[0] <= cx <= gt[0] + gt[2] and gt[1] <= cy <= gt[1] + gt[3]:
                    hits += 1
    out = {
        "clip": name,
        "pct_pixels_active": round(100 * float(np.mean(area_frac)), 1),
    }
    if name in ("hud_pulse", "pure_pan_no_object"):
        out["hud_energy_share_pct"] = round(100 * float(np.sum(hud_e) / np.sum(tot_e)), 2)
    if boxes[0] is not None:
        out["target_energy_share_pct"] = round(100 * float(np.mean(share)), 1)
        out["top_region_hits_target_pct"] = round(100 * hits / (N - 1), 1)
    return out, frames


def two_mover_split(axis_pair):
    """Energy at each mover, from the horizontal_vs_vertical / dim_vs_bright clips."""
    name, a_box_fn, b_box_fn = axis_pair
    frames, _ = make_clip(name)
    a, b = [], []
    for t in range(N - 1):
        _, energy, _ = detect_motion_region(frames[t], frames[t + 1])
        a.append(region_energy(energy, a_box_fn(t)))
        b.append(region_energy(energy, b_box_fn(t)))
    return float(np.sum(a)), float(np.sum(b))


if __name__ == "__main__":
    results = []
    for name in ["slow_explore", "fast_combat", "hud_pulse", "pure_pan_no_object"]:
        r, _ = run(name)
        results.append(r)
        print(r)

    hv_a, hv_b = two_mover_split((
        "horizontal_vs_vertical",
        lambda t: (max(160 + t * 4 - 40, 0), 80, 80, 80),
        lambda t: (440, max(60 + t * 4 - 40, 0), 80, 80)))
    print({"horizontal_vs_vertical_energy_ratio_H_over_V": round(hv_a / max(hv_b, 1e-9), 1)})

    db_a, db_b = two_mover_split((
        "dim_vs_bright",
        lambda t: (max(160 + t * 4 - 40, 0), 80, 80, 80),
        lambda t: (max(160 + t * 4 - 40, 0), 220, 80, 80)))
    print({"bright_over_dim_energy_ratio": round(db_a / max(db_b, 1e-9), 1)})

    json.dump({"clips": results,
               "h_over_v_ratio": round(hv_a / max(hv_b, 1e-9), 1),
               "bright_over_dim_ratio": round(db_a / max(db_b, 1e-9), 1)},
              open("game_motion_results.json", "w"), indent=2)
