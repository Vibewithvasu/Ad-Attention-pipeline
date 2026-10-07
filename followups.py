import json
import cv2
import numpy as np
from motion_emd import reichardt_motion_map, detect_motion_region
import game_motion_test as g

H, W, N = g.H, g.W, g.N
res = {}

# ---- (a) does a bright mover collapse everything else via per-frame min-max normalisation?
def raw_stats(frames):
    ratios, act = [], []
    for t in range(N - 1):
        f0, f1 = frames[t].astype(np.float32), frames[t + 1].astype(np.float32)
        d = np.abs(f1 * np.roll(f0, -1, 1) - np.roll(f1, -1, 1) * f0)
        ratios.append(float(d.max() / (np.median(d) + 1e-9)))
        e, _ = reichardt_motion_map(frames[t], frames[t + 1])
        act.append(float((e > 15).mean()))
    return float(np.median(ratios)), 100 * float(np.mean(act))

pan_only, _ = g.make_clip("pure_pan_no_object")
fight, _ = g.make_clip("fast_combat")
# same fast-combat pan but WITHOUT the bright fighters
nofight = []
for t in range(N):
    nofight.append(g.crop(g.WORLD, 200 + t * 6))
res["a_normalisation"] = {
    "pan_only_fast_active_pct": round(raw_stats(nofight)[1], 1),
    "pan_plus_bright_fighters_active_pct": round(raw_stats(fight)[1], 1),
}

# ---- (b) direction blindness: thin HORIZONTAL bar moving vertically vs thin VERTICAL bar moving horizontally
flat = np.full((H, W), 60, np.uint8)
def bar_clip(kind):
    fr = []
    for t in range(N):
        f = flat.copy()
        if kind == "hbar_moves_vertical":
            y = 40 + t * 4
            cv2.rectangle(f, (100, y), (500, y + 6), 250, -1)
        else:
            x = 40 + t * 4
            cv2.rectangle(f, (x, 80), (x + 6, 280), 250, -1)
        fr.append(f)
    return fr

def total_energy(frames):
    s = 0.0
    for t in range(N - 1):
        d = np.abs(frames[t + 1].astype(np.float32) * np.roll(frames[t].astype(np.float32), -1, 1)
                   - np.roll(frames[t + 1].astype(np.float32), -1, 1) * frames[t].astype(np.float32))
        s += float(d.sum())
    return s
ev = total_energy(bar_clip("hbar_moves_vertical"))
eh = total_energy(bar_clip("vbar_moves_horizontal"))
res["b_direction"] = {"vbar_horizontal_motion_over_hbar_vertical_motion": round(eh / max(ev, 1e-9), 1)}

# ---- (c) same contrast (+100), different absolute brightness
def contrast_clip(bg, fg):
    fr = []
    for t in range(N):
        f = np.full((H, W), bg, np.uint8)
        cv2.circle(f, (100 + t * 4, 180), 20, fg, -1)
        fr.append(f)
    return fr
dark = total_energy(contrast_clip(20, 120))
bright = total_energy(contrast_clip(130, 230))
res["c_brightness"] = {"equal_contrast_bright_over_dark_energy": round(bright / max(dark, 1e-9), 1)}

# ---- (d) hud_pulse: energy of pulsing dot vs slow mover vs HUD
frames, _ = g.make_clip("hud_pulse")
pulse_e, move_e, hud_e = 0.0, 0.0, 0.0
for t in range(N - 1):
    _, e, _ = detect_motion_region(frames[t], frames[t + 1])
    pulse_e += float(e[220:300, 480:560].sum())
    cx = 320 + t * 3
    move_e += float(e[130:190, max(cx - 25, 0):cx + 25].sum())
    hud_e += float(e[20:44, 20:220].sum() + e[262:338, 32:108].sum())
tot = pulse_e + move_e + hud_e + 1e-9
res["d_hud_pulse"] = {
    "pulsing_dot_share_pct": round(100 * pulse_e / tot, 1),
    "slow_mover_share_pct": round(100 * move_e / tot, 1),
    "static_hud_share_pct": round(100 * hud_e / tot, 1),
}

print(json.dumps(res, indent=2))
json.dump(res, open("followup_results.json", "w"), indent=2)
