"""
saliency_v2.py
==============
Upgrade of the static-image saliency stage, trained and validated against REAL
human eye-tracking data (OSIE, Xu et al. 2014, MIT licence).

What changed vs saliency_pipeline.compute_heatmap (v1):

  1. COLOUR FIX. v1 used only Spectral Residual + Fine-Grained saliency, which
     respond to edges / spatial frequency and under-score flat colour blocks
     (the CTA-button blind spot documented in NOTES.md section 8). v2 adds:
        - Frequency-Tuned colour saliency   (Achanta et al., CVPR 2009)
        - Colour-opponent centre-surround   (Itti, Koch & Niebur, 1998)
        - Luminance centre-surround
        - Local contrast energy
  2. MULTI-FACE prior (v1 used only the single best face).
  3. CENTRE prior -- the well-documented human tendency to look near the
     centre of a picture (Tatler, 2007).
  4. LEARNED FUSION. Instead of hand-set 0.5/0.5 and 0.4/0.6 weights, channel
     weights are fitted (non-negative least squares) on OSIE training images
     and scored on held-out test images that were never used for fitting.

Honest scope: OSIE is natural-scene free viewing, not advertising. Weights
should be re-fitted on ad-specific eye-tracking data if/when you can get it.
"""
import json
import os
import cv2
import numpy as np

WORK = (400, 300)      # working resolution (w, h) for feature extraction
OUT = (200, 150)       # scoring resolution (w, h)
CHANNELS = ["sr", "fg", "ft", "color_dog", "lum_dog", "contrast", "face", "center"]
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, "saliency_v2_weights.json")


def _z(m):
    m = m.astype(np.float32)
    return (m - m.mean()) / (m.std() + 1e-6)


def _dog(ch, s1, s2):
    return np.abs(cv2.GaussianBlur(ch, (0, 0), s1) - cv2.GaussianBlur(ch, (0, 0), s2))


def get_faces(img, net, conf_thresh=0.85):
    """All faces (NMS-merged), not just the best one. Returns list of (x,y,w,h)."""
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1 / 128.0, (320, 240), (127, 127, 127), swapRB=True)
    net.setInput(blob)
    scores, boxes = net.forward(["scores", "boxes"])
    scores, boxes = scores[0], boxes[0]
    cands, confs = [], []
    for i in range(scores.shape[0]):
        c = float(scores[i][1])
        if c > conf_thresh:
            x1, y1, x2, y2 = boxes[i]
            cands.append([int(x1 * w), int(y1 * h), int((x2 - x1) * w), int((y2 - y1) * h)])
            confs.append(c)
    if not cands:
        return []
    idx = cv2.dnn.NMSBoxes(cands, confs, conf_thresh, 0.3)
    return [tuple(cands[i]) for i in np.array(idx).flatten()]


def face_prior(shape_hw, faces):
    h, w = shape_hw
    out = np.zeros((h, w), np.float32)
    Y, X = np.ogrid[:h, :w]
    for (x, y, fw, fh) in faces:
        cx, cy = x + fw / 2, y + fh / 2
        r = max(fw, fh) * 0.8
        out = np.maximum(out, np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * r ** 2)).astype(np.float32))
    return out


def center_prior(shape_hw):
    h, w = shape_hw
    Y, X = np.ogrid[:h, :w]
    return np.exp(-(((X - w / 2) / (0.28 * w)) ** 2 + ((Y - h / 2) / (0.28 * h)) ** 2) / 2).astype(np.float32)


_SR = _FG = None


def compute_channels(img_bgr, face_net=None):
    """Return {channel_name: float32 map at OUT resolution}, each z-scored
    (except the two priors, which are kept in [0,1])."""
    global _SR, _FG
    if _SR is None:
        _SR = cv2.saliency.StaticSaliencySpectralResidual_create()
        _FG = cv2.saliency.StaticSaliencyFineGrained_create()
    img = cv2.resize(img_bgr, WORK, interpolation=cv2.INTER_AREA)
    ch = {}
    _, s = _SR.computeSaliency(img); ch["sr"] = s.astype(np.float32)
    _, s = _FG.computeSaliency(img); ch["fg"] = s.astype(np.float32)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    blur = cv2.GaussianBlur(lab, (0, 0), 2)
    mean = lab.reshape(-1, 3).mean(0)
    ch["ft"] = np.linalg.norm(blur - mean, axis=2)                       # Achanta 2009
    ch["color_dog"] = sum(_dog(lab[:, :, c], s1, s2) for c in (1, 2) for s1, s2 in ((2, 8), (4, 16), (8, 32)))
    ch["lum_dog"] = sum(_dog(lab[:, :, 0], s1, s2) for s1, s2 in ((2, 8), (4, 16), (8, 32)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    ch["contrast"] = cv2.GaussianBlur(np.sqrt(gx ** 2 + gy ** 2), (0, 0), 6)

    faces = get_faces(img_bgr, face_net) if face_net is not None else []
    fp = face_prior(img_bgr.shape[:2], faces)
    ch["face"] = cv2.resize(fp, OUT, interpolation=cv2.INTER_AREA)
    ch["center"] = center_prior((OUT[1], OUT[0]))
    for k in ("sr", "fg", "ft", "color_dog", "lum_dog", "contrast"):
        ch[k] = _z(cv2.resize(ch[k], OUT, interpolation=cv2.INTER_AREA))
    return ch


def load_weights(path=WEIGHTS_PATH):
    with open(path) as f:
        return json.load(f)


def predict(channels, weights=None):
    """Fuse channels with learned weights -> saliency map at OUT resolution."""
    w = weights or load_weights()
    m = sum(w["weights"][k] * channels[k] for k in CHANNELS)
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), w["blur_sigma"])
    return m - m.min()


def compute_heatmap_v2(img_bgr, face_net=None, weights=None):
    """Drop-in analogue of compute_heatmap(): returns uint8 heatmap at the
    ORIGINAL image resolution, ready for aoi_attention_share()."""
    sm = predict(compute_channels(img_bgr, face_net), weights)
    h, w = img_bgr.shape[:2]
    sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_CUBIC)
    sm = cv2.GaussianBlur(sm, (0, 0), max(h, w) / 160)
    return cv2.normalize(np.clip(sm, 0, None), None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
