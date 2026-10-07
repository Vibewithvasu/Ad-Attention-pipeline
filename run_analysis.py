"""
run_analysis.py
================
End-to-end example: runs the full saliency_pipeline on one ad creative and
produces every output discussed in the project (heatmap, AOI attention
shares, Focus/concentration score, competition flags, and the AOI-level
gaze path with a synergy report).

USAGE:
    python3 run_analysis.py path/to/banner.png

    You will need to define the AOI boxes for your specific creative --
    see the AOIS list below for the pattern. In the full project these were
    built semi-automatically:
        - Face box:        detected automatically (get_face_box)
        - Text boxes:       detected automatically via OCR (get_ocr_lines),
                             then grouped into logical AOIs by hand
                             (e.g. "headline", "offer amount", "CTA")
        - Logo/graphic boxes: annotated manually by inspecting the image,
                             since these are graphical, not text

MODELS REQUIRED (see download_models.py to fetch them):
    models/face_detector2.onnx   -- UltraFace RFB-320 face detector
"""
import sys
import cv2
import numpy as np
from saliency_v2 import compute_heatmap_v2
from saliency_pipeline import (
    get_ocr_lines, get_face_box, compute_heatmap, aoi_attention_share,
    focus_score, detect_competition_relative, aoi_gaze_path,
    draw_gaze_path, synergy_report
)

def analyze(image_path, aois=None, out_prefix="analysis", version="v2"):
    """version="v2" (default): learned-fusion saliency validated on OSIE human eye-tracking.
    version="v1": the original Spectral-Residual + Fine-Grained + single-face model."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]
    print(f"Loaded {image_path}  ({w}x{h})")

    # --- 1. Face detection (attention prior) ---
    net = cv2.dnn.readNetFromONNX("models/face_detector2.onnx")
    face_box = get_face_box(img, net)
    print("Face box:", face_box)

    # --- 2. OCR pass -- useful for building/checking your AOI boxes ---
    print("\nOCR-detected text lines (use these to define your text AOIs):")
    for line in get_ocr_lines(img):
        print(" ", line)

    # --- 3. If the caller didn't pass AOIs, bail out with guidance ---
    if aois is None:
        print("\nNo AOIs supplied -- define them using the OCR output above "
              "plus manual boxes for logos/graphics, then re-run with aois=[...].")
        return None

    for a in aois:
        if a["box"] is None:
            a["box"] = face_box

    # --- 4. Bottom-up saliency + face-prior heatmap ---
    saliency_sr = cv2.saliency.StaticSaliencySpectralResidual_create()
    saliency_fg = cv2.saliency.StaticSaliencyFineGrained_create()
    if version == "v1":
        heatmap = compute_heatmap(img, saliency_sr, saliency_fg, face_box)
    else:
        heatmap = compute_heatmap_v2(img, net)
    print(f"Saliency model: {version}")

    # --- 5. Per-AOI attention share (mirrors "Total Attention" in commercial tools) ---
    shares = aoi_attention_share(heatmap, aois)
    print("\nAOI attention shares:")
    for s in shares:
        print(f"  {s['label']:<40} {s['share']:.1f}%")

    # --- 6. Focus/concentration score (entropy-based) ---
    fscore = focus_score(shares)
    unassigned = 100 - sum(s["share"] for s in shares)
    print(f"\nFocus score (0=fragmented, 1=concentrated): {fscore:.3f}")
    print(f"Unassigned attention (background/edges, no AOI): {unassigned:.1f}%")

    # --- 7. Competition detector ---
    flags = detect_competition_relative(shares, img_diag=np.sqrt(w**2 + h**2))
    print("\nCompetition flags (near-tied AOIs that are spatially close):")
    if not flags:
        print("  None found.")
    for f in flags:
        print(f"  {f['a']} ({f['a_share']:.1f}%) vs {f['b']} ({f['b_share']:.1f}%), {f['dist']:.0f}px apart")

    # --- 8. AOI-level gaze path + synergy report ---
    path = aoi_gaze_path(shares)
    lines, gaps = synergy_report(path)
    print("\nPredicted gaze sequence (by attention-share priority):")
    for l in lines:
        print(l)
    print("\nGap between consecutive fixations (small gap = competition, large = clear hierarchy):")
    for g in gaps:
        flag = "  <-- COMPETING (near-tie)" if abs(g[2]) < 2.0 else ""
        print(f"  {g[0]} -> {g[1]}: {g[2]}pt{flag}")

    # --- 9. Save the combined visualization ---
    overlay = draw_gaze_path(img, heatmap, shares, path)
    out_path = f"{out_prefix}_gazepath.png"
    cv2.imwrite(out_path, overlay)
    print(f"\nSaved combined heatmap + gaze path visualization -> {out_path}")

    return {
        "shares": shares, "focus_score": fscore, "unassigned": unassigned,
        "competition_flags": flags, "gaze_path": path,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Example AOI set -- EDIT THIS to match your creative's actual layout.
    # box=None means "use the auto-detected face box".
    example_aois = [
        {"label": "Brand Logo", "box": (30, 40, 280, 60)},
        {"label": "Bank Logo", "box": (550, 40, 290, 55)},
        {"label": "Headline", "box": (40, 190, 460, 145)},
        {"label": "Offer Amount", "box": (35, 440, 800, 445)},
        {"label": "Cashback line", "box": (40, 880, 420, 100)},
        {"label": "Face", "box": None},
        {"label": "Supporting graphics", "box": (30, 1440, 260, 210)},
        {"label": "CTA button", "box": (25, 1665, 800, 110)},
    ]

    ver = "v1" if "--v1" in sys.argv else "v2"
    analyze(sys.argv[1], aois=example_aois, out_prefix=f"analysis_{ver}", version=ver)
