"""Re-run of the NOTES.md section-8 blind spot (flat vs bordered CTA) plus an equiluminant colour pop-out probe.
Each stimulus is analysed in isolation; metric = share of total heat inside the button box."""
import json, cv2, numpy as np
from saliency_pipeline import compute_heatmap
import saliency_v2 as V2

sr = cv2.saliency.StaticSaliencySpectralResidual_create()
fg = cv2.saliency.StaticSaliencyFineGrained_create()
H, W = 600, 800
BOX = (280, 260, 240, 80)

def canvas(bg=235):
    return np.full((H, W, 3), bg, np.uint8)

def button(img, fill, border=None):
    x, y, w, h = BOX
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1)
    if border is not None:
        cv2.rectangle(img, (x, y), (x + w, y + h), border, 6)
    return img

def share(heat):
    x, y, w, h = BOX
    return float(heat[y:y + h, x:x + w].sum() / (heat.sum() + 1e-9) * 100)

def both(img):
    return {"v1": share(compute_heatmap(img, sr, fg, None)), "v2": share(V2.compute_heatmap_v2(img, None))}

res = {}
red = (60, 60, 200)
lab = cv2.cvtColor(np.uint8([[red]]), cv2.COLOR_BGR2LAB)[0, 0]
# grey with the same CIE lightness as the red
g = int(np.argmin([abs(int(cv2.cvtColor(np.uint8([[[v, v, v]]]), cv2.COLOR_BGR2LAB)[0, 0, 0]) - int(lab[0])) for v in range(256)]))
res["flat_red_button"] = both(button(canvas(), red))
res["bordered_red_button"] = both(button(canvas(), red, (20, 20, 20)))
res["equiluminant_red_on_grey"] = both(button(canvas(g), red))
res["equiluminant_grey_control_(no_colour_contrast)"] = both(button(canvas(g), (g, g, g)))
res["empty_page_(no_button)"] = both(canvas())
for k, v in res.items():
    print(f"{k:<48} v1 {v['v1']:6.2f}%   v2 {v['v2']:6.2f}%")
fl, bd = res["flat_red_button"], res["bordered_red_button"]
res["bordered_over_flat_ratio"] = {"v1": round(bd["v1"] / fl["v1"], 2), "v2": round(bd["v2"] / fl["v2"], 2)}
print("bordered/flat ratio  (1.0 = no edge bias):", res["bordered_over_flat_ratio"])
json.dump(res, open("button_test_results.json", "w"), indent=2)
