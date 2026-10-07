"""
download_models.py
===================
Fetches the one model file the pipeline needs: UltraFace RFB-320, an
ONNX face detector (~1.2MB, CC BY 4.0, github.com/Linzaer/Ultra-Light-Fast-
Generic-Face-Detector-1MB). Run this once before run_analysis.py.

Why this specific model: OpenCV's built-in Haar cascade face detector was
tried first and produced false positives on cash/phone-screen texture
(see NOTES.md, "Debugging journey"). A Caffe-based DNN detector was tried
next but failed because current OpenCV builds have removed the Caffe
importer. UltraFace (ONNX format) was the working substitute.
"""
import os
import urllib.request

MODEL_URL = "https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/master/models/onnx/version-RFB-320.onnx"
MODEL_PATH = "models/face_detector2.onnx"

def main():
    os.makedirs("models", exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print(f"Already present: {MODEL_PATH}")
        return
    print(f"Downloading {MODEL_URL} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"Saved to {MODEL_PATH} ({size_kb:.0f} KB)")

if __name__ == "__main__":
    main()
