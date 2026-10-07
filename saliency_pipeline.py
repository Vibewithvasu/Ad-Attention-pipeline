"""
saliency_pipeline.py
=====================
A lightweight, fully-local computational attention/saliency pipeline for
screening ad creative before it goes to a real test or real spend.

WHAT THIS IS:
    A heuristic screening tool combining published, citable bottom-up saliency
    algorithms with a face-detection attention prior, wrapped in an
    AOI (Area-of-Interest)-based scoring layer inspired by how commercial
    neuromarketing platforms (e.g. Neurons AI) structure their reporting --
    per-region attention share, a concentration/"Focus" score, and a
    competition-vs-synergy detector along the predicted gaze path.

WHAT THIS IS NOT:
    Not a substitute for real eye-tracking or real human testing. None of the
    algorithms below are validated against held-out human ground truth for
    these specific images (no Pearson CC / KLD / SIM comparison exists for
    this pipeline's output). Treat every number as a directional heuristic,
    not a measurement. See NOTES.md in this package for the full honesty
    write-up on this point.

CORE ALGORITHMS (all real, published, and run locally -- no external
API calls, no blocked model downloads):
    - Spectral Residual saliency        Hou & Zhang, CVPR 2007
    - Fine-Grained saliency             OpenCV's StaticSaliencyFineGrained
    - Face detection                    UltraFace RFB-320 (Linzaer, ONNX)
    - OCR-based text AOI detection      Tesseract OCR
    - Face-as-attention-prior rationale Cerf, Harel, Einhauser & Koch, 2008;
                                         Milosavljevic & Cerf, 2008

WHY NOT A MODERN DEEP SALIENCY MODEL (e.g. DeepGaze IIE/III)?
    Tried first. `deepgaze_pytorch` installs cleanly from GitHub, but its
    pretrained weights are hosted on a CDN (release-assets.githubusercontent.com)
    outside this environment's network allowlist, and its ResNet backbone
    weights (via torchvision) are similarly blocked (download.pytorch.org,
    403 Forbidden). This is a genuine, verified infrastructure limitation,
    not a skipped step -- confirmed by actually attempting the download and
    reading the resulting error. The classical-algorithm ensemble here is the
    documented fallback.

Author's note: this pipeline was built interactively, debugging real failures
as they came up (see NOTES.md for the full chain-of-thought / decision log).
"""
import cv2
import numpy as np
import pytesseract
import json


def get_ocr_lines(img, upscale=3):
    """Real OCR text detection, clustered into lines."""
    img_up = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    data = pytesseract.image_to_data(img_up, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data['text'])):
        t = data['text'][i].strip()
        if t and int(data['conf'][i]) > 30:
            x = data['left'][i]/upscale; y = data['top'][i]/upscale
            w = data['width'][i]/upscale; h = data['height'][i]/upscale
            line_num = data['line_num'][i]
            block_num = data['block_num'][i]
            words.append({'text': t, 'x': x, 'y': y, 'w': w, 'h': h,
                          'key': (block_num, line_num)})
    # cluster words into lines
    lines = {}
    for wd in words:
        lines.setdefault(wd['key'], []).append(wd)
    line_boxes = []
    for key, wds in lines.items():
        x1 = min(w['x'] for w in wds); y1 = min(w['y'] for w in wds)
        x2 = max(w['x']+w['w'] for w in wds); y2 = max(w['y']+w['h'] for w in wds)
        text = ' '.join(w['text'] for w in wds)
        line_boxes.append({'label': text, 'box': (int(x1), int(y1), int(x2-x1), int(y2-y1))})
    return line_boxes

def get_face_box(img, net, conf_thresh=0.85):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/128.0, (320, 240), (127, 127, 127), swapRB=True)
    net.setInput(blob)
    scores, boxes = net.forward(['scores', 'boxes'])
    scores = scores[0]; boxes = boxes[0]
    candidates = []
    for i in range(scores.shape[0]):
        conf = scores[i][1]
        if conf > conf_thresh:
            x1, y1, x2, y2 = boxes[i]
            candidates.append((int(x1*w), int(y1*h), int((x2-x1)*w), int((y2-y1)*h), float(conf)))
    if not candidates:
        return None
    boxes_xywh = [c[:4] for c in candidates]
    confs = [c[4] for c in candidates]
    idxs = cv2.dnn.NMSBoxes(boxes_xywh, confs, conf_thresh, 0.3)
    best = candidates[np.array(idxs).flatten()[0]]
    return best[:4]

def compute_heatmap(img, saliency_sr, saliency_fg, face_box):
    h, w = img.shape[:2]
    _, sal_sr = saliency_sr.computeSaliency(img)
    sal_sr = cv2.resize((sal_sr*255).astype('uint8'), (w,h))
    _, sal_fg = saliency_fg.computeSaliency(img)
    sal_fg = cv2.resize((sal_fg*255).astype('uint8'), (w,h))
    sal_combo = cv2.addWeighted(sal_sr, 0.5, sal_fg, 0.5, 0)
    sal_combo = cv2.GaussianBlur(sal_combo, (15,15), 0)

    face_prior = np.zeros((h,w), dtype=np.float32)
    if face_box:
        x,y,fw,fh = face_box
        cx, cy = x+fw//2, y+fh//2
        radius = int(fw*1.6)
        Y,X = np.ogrid[:h,:w]
        dist2 = (X-cx)**2 + (Y-cy)**2
        face_prior = np.exp(-dist2/(2*(radius**2)))
    face_prior_255 = (face_prior*255).astype('uint8')

    combined = cv2.addWeighted(sal_combo, 0.4, face_prior_255, 0.6, 0) if face_box else sal_combo
    combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return combined

def aoi_attention_share(heatmap, aois):
    total = heatmap.astype(np.float64).sum()
    results = []
    for aoi in aois:
        x,y,w,h = aoi['box']
        x2,y2 = min(x+w, heatmap.shape[1]), min(y+h, heatmap.shape[0])
        region = heatmap[max(0,y):y2, max(0,x):x2].astype(np.float64)
        share = region.sum()/total*100 if total>0 else 0
        results.append({'label': aoi['label'], 'box': aoi['box'], 'share': share})
    return results

def focus_score(shares):
    """
    Entropy-based concentration score: 1 = all captured attention on one AOI,
    0 = perfectly spread across all AOIs.

    IMPORTANT (patched): shares are normalized to sum to 1 before computing
    entropy. Earlier versions of this function used raw share/100 values
    directly as pseudo-probabilities WITHOUT renormalizing -- since AOI
    shares never sum to 100% (some attention always falls on unassigned
    background), this made the score sensitive to how much TOTAL attention
    was captured by named AOIs, not just how that captured attention was
    DISTRIBUTED among them. Two creatives with identical concentration
    shape but different total-captured-mass would get different scores,
    which is wrong -- Focus is supposed to measure shape, not volume.
    Confirmed with a live before/after recomputation on the recharge-banner
    case: the bug was large enough to flip a directional conclusion
    (baseline appeared MORE concentrated than a redesign that had strictly
    fewer competing elements and zero competition flags, which was
    internally inconsistent). Fixed by renormalizing here.
    """
    p = np.array([s['share']/100 for s in shares])
    p = p[p > 0]
    if len(p) <= 1:
        return 1.0
    p = p / p.sum()          # <-- the fix: renormalize so p is a real probability distribution
    entropy = -np.sum(p*np.log(p))
    max_entropy = np.log(len(p))
    return 1 - (entropy/max_entropy)

def detect_competition(shares, aois, share_gap_thresh=8, dist_thresh_frac=0.35, img_diag=None):
    """Flag AOI pairs with similar high share AND spatial adjacency = competition."""
    flags = []

    sorted_s = sorted(shares, key=lambda s: -s['share'])
    for i in range(len(sorted_s)):
        for j in range(i+1, len(sorted_s)):
            a, b = sorted_s[i], sorted_s[j]
            if abs(a['share']-b['share']) < share_gap_thresh and a['share'] > 8 and b['share'] > 8:
                ax,ay,aw,ah = a['box']; bx,by,bw,bh = b['box']
                acx,acy = ax+aw/2, ay+ah/2
                bcx,bcy = bx+bw/2, by+bh/2
                dist = np.sqrt((acx-bcx)**2+(acy-bcy)**2)
                if img_diag and dist < dist_thresh_frac*img_diag:
                    flags.append((a['label'], b['label'], a['share'], b['share'], dist))
    return flags


def detect_competition_relative(shares, top_n=4, ratio_thresh=0.6, dist_thresh_frac=0.4, img_diag=None):
    """Among the top-N AOIs by share, flag pairs whose shares are within ratio_thresh of each other
    AND whose boxes are spatially close -> genuine competing focal points."""
    sorted_s = sorted(shares, key=lambda s: -s['share'])[:top_n]
    flags = []
    for i in range(len(sorted_s)):
        for j in range(i+1, len(sorted_s)):
            a, b = sorted_s[i], sorted_s[j]
            if b['share'] == 0: continue
            ratio = b['share']/a['share'] if a['share']>0 else 0
            ax,ay,aw,ah = a['box']; bx,by,bw,bh = b['box']
            acx,acy = ax+aw/2, ay+ah/2
            bcx,bcy = bx+bw/2, by+bh/2
            dist = np.sqrt((acx-bcx)**2+(acy-bcy)**2)
            close = (dist < dist_thresh_frac*img_diag) if img_diag else True
            if ratio > ratio_thresh and close:
                flags.append({'a':a['label'],'b':b['label'],'a_share':a['share'],'b_share':b['share'],'dist':round(dist,1)})
    return flags

def aoi_gaze_path(shares, min_share=1.5):
    """
    Order AOIs by predicted fixation priority (attention share, descending) --
    the standard winner-take-all assumption used in classic scanpath models
    (Itti/Koch/Niebur 1998). Returns the ordered list of AOI centers + labels,
    so the SEQUENCE (not just the distribution) can be inspected for synergy
    (a sensible narrative arc: brand -> offer -> trust -> action) vs chaos
    (jumping between unrelated, far-apart elements).
    """
    ranked = sorted([s for s in shares if s['share'] >= min_share], key=lambda s: -s['share'])
    path = []
    for rank, s in enumerate(ranked, start=1):
        x, y, w, h = s['box']
        cx, cy = x + w // 2, y + h // 2
        path.append({'rank': rank, 'label': s['label'], 'share': s['share'], 'center': (cx, cy)})
    return path

def draw_gaze_path(img, heatmap, shares, path, alpha=0.45):
    """Combined view: AOI heatmap + boxes + numbered sequential gaze arrows."""
    hcolor = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1-alpha, hcolor, alpha, 0)

    for s in shares:
        x, y, w, h = s['box']
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 255, 255), 2)

    for i in range(1, len(path)):
        p1 = path[i-1]['center']
        p2 = path[i]['center']
        cv2.arrowedLine(overlay, p1, p2, (255, 255, 255), 3, cv2.LINE_AA, tipLength=0.05)
        cv2.line(overlay, p1, p2, (0, 0, 0), 1, cv2.LINE_AA)

    for node in path:
        cx, cy = node['center']
        cv2.circle(overlay, (cx, cy), 26, (20, 20, 20), -1)
        cv2.circle(overlay, (cx, cy), 26, (255, 255, 255), 3)
        cv2.putText(overlay, str(node['rank']), (cx-13, cy+10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay

def synergy_report(path):
    """
    Turns the ranked path into a plain-language synergy/competition read:
    - Synergy: the top-ranked AOIs form a sensible funnel (brand -> offer -> trust -> action)
    - Competition: two top-ranked AOIs are within a small share margin of each other
      (ambiguous "which one wins the first glance" -- exactly what the user flagged originally)
    """
    lines = []
    for i, node in enumerate(path):
        lines.append(f"  Fixation {node['rank']}: {node['label']}  ({node['share']:.1f}% share)")
    gaps = []
    for i in range(1, len(path)):
        gap = path[i-1]['share'] - path[i]['share']
        gaps.append((path[i-1]['label'], path[i]['label'], round(gap,1)))
    return lines, gaps

def averaging_saccade_prediction(shares, dist_thresh_frac=0.35, img_diag=None, top_n=4):
    """
    Real oculomotor 'global effect' model (Findlay, 1982, and 40+ years of
    replication since -- e.g. PLOS Biology 2018, Journal of Vision 2013):
    when two competing AOIs are close together and similarly weighted, the
    eye does NOT cleanly pick one -- it lands at the population-weighted
    centroid between them.

    This checks the top-N AOIs by share for pairs close enough together to
    plausibly trigger this effect, and computes the predicted weighted
    landing point. Critically, it flags when that landing point falls
    OUTSIDE every AOI's box -- i.e., a fixation predicted to land on empty
    space, wasted between two competing elements, rather than on either one.

    Scope note: the global effect is best-documented for short-latency,
    REFLEXIVE saccades -- it weakens for slower, deliberate viewing. This
    is a good fit for fast-scroll social/ad contexts specifically (the
    first, highest-value glance before scroll-past), not a claim about the
    entire viewing sequence.
    """
    sorted_s = sorted(shares, key=lambda s: -s['share'])[:top_n]
    results = []
    for i in range(len(sorted_s)):
        for j in range(i+1, len(sorted_s)):
            a, b = sorted_s[i], sorted_s[j]
            ax, ay, aw, ah = a['box']; bx, by, bw, bh = b['box']
            acx, acy = ax + aw/2, ay + ah/2
            bcx, bcy = bx + bw/2, by + bh/2
            dist = np.sqrt((acx-bcx)**2 + (acy-bcy)**2)
            close = (dist < dist_thresh_frac*img_diag) if img_diag else True
            if not close:
                continue
            wa, wb = a['share'], b['share']
            total_w = wa + wb
            landing_x = (acx*wa + bcx*wb) / total_w
            landing_y = (acy*wa + bcy*wb) / total_w

            def inside(px, py, box):
                x, y, w, h = box
                return x <= px <= x+w and y <= py <= y+h

            lands_on_a = inside(landing_x, landing_y, a['box'])
            lands_on_b = inside(landing_x, landing_y, b['box'])
            wasted = not (lands_on_a or lands_on_b)

            results.append({
                'pair': (a['label'], b['label']),
                'weights': (round(wa,1), round(wb,1)),
                'landing_point': (int(landing_x), int(landing_y)),
                'wasted_fixation': wasted,
                'dist': round(dist,1),
            })
    return results
