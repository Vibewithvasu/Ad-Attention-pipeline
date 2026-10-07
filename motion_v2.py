"""
motion_v2.py
============
Rebuild of the motion-salience module after the stress test (see NOTES.md
section 16). Fixes every flaw that test exposed in motion_emd.py:

  FLAW 1  per-frame min-max normalisation let one bright mover erase all other
          motion                     -> FIXED-SCALE energy (absolute units)
  FLAW 2  horizontal-only correlator, ~130x blind to vertical motion of
          horizontal edges           -> BIDIRECTIONAL (x and y) correlators
  FLAW 3  brightness inflated the signal (same contrast, 2.6x energy)
                                     -> LOCAL CONTRAST NORMALISATION, motivated
          by contrast normalisation in fly motion vision (Drews et al., 2020,
          Current Biology 30:209) and divisive normalisation generally
  FLAW 4  a camera pan lit up the whole frame (47% of pixels)
                                     -> EGO-MOTION COMPENSATION: global motion is
          estimated robustly (LK + RANSAC similarity) and cancelled, so only
          independently moving things remain. Guarded so a lone moving object on
          a flat background is never mistaken for camera motion.
  FLAW 5  only the single largest motion region was returned
                                     -> TOP-K regions with energy shares
  EXTRA   a single frame-to-frame offset is tuned to ~1 px/frame, so fast
          motion fell between detectors -> MULTI-SCALE pyramid (speed tuning
          doubles at every level)

Core detector is still the Hassenstein-Reichardt correlator (1956).
Honest scope: validated on synthetic clips with known ground truth and on a
background-subtraction proxy for real video. NOT validated against human
eye-tracking of video (no open video eye-tracking set was reachable).
"""
import cv2
import numpy as np

SIGMA_PRE = 1.0      # spatial pre-blur (px) before correlating
SIGMA_LOC = 8.0      # surround size for local contrast normalisation (px)
CONTRAST_FLOOR = 6.0 # grey levels; contrast below this is not amplified
LEVELS = 3           # pyramid levels (full, 1/2, 1/4)
COMPRESS_K = 0.35    # log-compression constant ("signal compression")
DEFAULT_THRESH = 0.54  # noise-floor threshold from calibrate_threshold() at 640x360, sigma=3 sensor noise


def _gray(f):
    g = f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return g if g.dtype == np.uint8 else np.clip(g, 0, 255).astype(np.uint8)


def _prep(g):
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), SIGMA_PRE)
    mu = cv2.GaussianBlur(g, (0, 0), SIGMA_LOC)
    d = g - mu
    sd = np.sqrt(cv2.GaussianBlur(d * d, (0, 0), SIGMA_LOC))
    return d / (sd + CONTRAST_FLOOR)


def _hr(a0, a1, axis):
    """Hassenstein-Reichardt opponent output for neighbouring-pixel pairs along `axis`."""
    b0 = np.roll(a0, -1, axis=axis)
    b1 = np.roll(a1, -1, axis=axis)
    return a1 * b0 - b1 * a0


def estimate_ego(g0, g1, min_pts=40, min_inlier_frac=0.4, min_spread=0.45):
    """Global (camera) motion as a 2x3 similarity matrix, or None.
    Accepts a global model only if it is supported by many tracked points spread
    over the frame; otherwise assumes a static camera."""
    h, w = g0.shape
    # corners first; if the scene is smooth (few corners) add a regular grid so a
    # global model can still be fitted. RANSAC discards the unreliable grid points.
    pts = cv2.goodFeaturesToTrack(g0, 500, 0.001, 8)
    gx, gy = np.meshgrid(np.linspace(24, w - 24, 24), np.linspace(24, h - 24, 14))
    grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32).reshape(-1, 1, 2)
    pts = grid if pts is None else np.concatenate([pts, grid], 0)
    nxt, st, err = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None, winSize=(21, 21), maxLevel=4)
    ok = (st.ravel() == 1) & (err.ravel() < 25)
    p0, p1 = pts[ok], nxt[ok]
    if len(p0) < min_pts:
        return None
    M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None:
        return None
    inl = inl.ravel().astype(bool)
    if inl.mean() < min_inlier_frac or inl.sum() < min_pts:
        return None
    pin = p0[inl].reshape(-1, 2)
    spread = ((pin[:, 0].max() - pin[:, 0].min()) / w) * ((pin[:, 1].max() - pin[:, 1].min()) / h)
    if spread < min_spread:
        return None
    return M


def _is_identity(M, tol_px=0.3, tol_s=0.002):
    return abs(M[0, 2]) < tol_px and abs(M[1, 2]) < tol_px and abs(M[0, 0] - 1) < tol_s and abs(M[1, 0]) < tol_s


def motion_saliency(f0, f1, compensate=True):
    """Returns dict:
        energy : float32 HxW, fixed-scale motion energy (comparable across frames/clips)
        ego    : None (static camera / not estimable) or dict(tx, ty, scale, rot_deg)
    """
    g0, g1 = _gray(f0), _gray(f1)
    h, w = g0.shape
    valid = np.ones((h, w), np.uint8)
    ego = None
    if compensate:
        M = estimate_ego(g0, g1)
        if M is not None and not _is_identity(M):
            ego = dict(tx=float(M[0, 2]), ty=float(M[1, 2]),
                       scale=float(np.hypot(M[0, 0], M[1, 0])),
                       rot_deg=float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))))
            # warp t0 onto t1's coordinates -> residual motion is independent motion
            g0 = cv2.warpAffine(g0, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            valid = cv2.warpAffine(np.ones((h, w), np.uint8), M, (w, h), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
    a0, a1 = _prep(g0), _prep(g1)
    energy = np.zeros((h, w), np.float32)
    for lv in range(LEVELS):
        if lv > 0:
            a0, a1 = cv2.pyrDown(a0), cv2.pyrDown(a1)
        ex = _hr(a0, a1, 1)
        ey = _hr(a0, a1, 0)
        e = np.sqrt(ex * ex + ey * ey)                       # direction-agnostic magnitude
        e = cv2.GaussianBlur(e, (0, 0), 1.5)
        if lv > 0:
            e = cv2.resize(e, (w, h), interpolation=cv2.INTER_LINEAR)
        energy = np.maximum(energy, e)
    energy = np.log1p(energy / COMPRESS_K).astype(np.float32)   # signal compression
    energy *= valid
    return {"energy": energy, "ego": ego}


def frame_threshold(energy, base=DEFAULT_THRESH, k=4.0):
    """Per-frame robust threshold: never below the noise floor, but raised when a
    frame carries a diffuse residual (handheld shake, parallax, compression noise)
    so only LOCALISED motion peaks survive. Found necessary on real handheld footage:
    without it v2 flagged >20% of pixels in 8/249 frames of a moving-camera clip."""
    med = float(np.median(energy))
    mad = float(np.median(np.abs(energy - med))) * 1.4826
    return max(base, med + k * mad)


def detect_regions(energy, thresh=None, top_k=5, min_area=60):
    """Top-K connected motion regions with their share of total supra-threshold energy."""
    if thresh is None:
        thresh = frame_threshold(energy)
    mask = (energy > thresh).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total = float(energy[energy > thresh].sum()) + 1e-9
    regs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        m = (lab == i) & (energy > thresh)
        e = float(energy[m].sum())
        regs.append({"box": (int(x), int(y), int(w), int(h)), "energy": e,
                     "share": e / total, "peak": float(energy[m].max()) if m.any() else 0.0})
    regs.sort(key=lambda r: -r["energy"])
    return regs[:top_k]


def calibrate_threshold(h=360, w=640, noise_sigma=3.0, n=8, margin=2.0, seed=0):
    """Noise-floor calibration: static textured scene + sensor noise -> energy
    at the 99.9th percentile, times a safety margin."""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 3) * 160 + 40
    vals = []
    for _ in range(n):
        f0 = np.clip(base + rng.normal(0, noise_sigma, base.shape), 0, 255).astype(np.uint8)
        f1 = np.clip(base + rng.normal(0, noise_sigma, base.shape), 0, 255).astype(np.uint8)
        vals.append(np.percentile(motion_saliency(f0, f1, compensate=False)["energy"], 99.9))
    return float(np.mean(vals) * margin)


def clip_motion_map(frames, compensate=True, step=1):
    """Whole-clip analysis: mean energy map, plus per-frame total energy
    (a motion-onset timeline) and the fraction of frames where the camera moved."""
    acc, per_frame, ego_frames = None, [], 0
    for t in range(0, len(frames) - 1, step):
        r = motion_saliency(frames[t], frames[t + 1], compensate)
        acc = r["energy"] if acc is None else acc + r["energy"]
        per_frame.append(float(r["energy"].sum()))
        ego_frames += int(r["ego"] is not None)
    n = max(len(per_frame), 1)
    return {"mean_energy": acc / n, "per_frame_total": per_frame, "camera_moving_frac": ego_frames / n}


def aoi_motion_share(energy, aois, thresh=0.0):
    """Share of motion energy inside each named AOI (same AOI dicts as the static pipeline)."""
    e = np.where(energy > thresh, energy, 0).astype(np.float64)
    total = e.sum() + 1e-12
    out = []
    for a in aois:
        x, y, w, h = a["box"]
        out.append({"label": a["label"], "box": a["box"],
                    "motion_share": float(e[max(y, 0):y + h, max(x, 0):x + w].sum() / total * 100)})
    return out
