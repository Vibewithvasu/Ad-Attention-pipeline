"""
motion_emd.py
=============
Elementary Motion Detector (EMD) -- Hassenstein-Reichardt correlator (1956)

A real, citable, biologically-inspired motion-detection algorithm, derived
from insect (beetle/fly) motion-perception experiments -- still used today
as legitimate front-end architecture for optic-flow/motion-salience
computation (see Borst & Egelhaaf; Franceschini).

This is a completely separate scientific lineage from the AOI/saliency
pipeline in saliency_pipeline.py -- it extends the project to a case that
pipeline was structurally blind to: VIDEO and animated creative (Reels,
animated banners, GIF ads), where a moving element can dominate attention
in ways a single static frame can never capture.

WHAT THIS IS NOT: not a fly-connectome simulation, not "neuroscience-
powered AI." It's one specific, validated correlator model, implemented
as a 2D array of detectors across two consecutive frames.
"""
import cv2
import numpy as np


def reichardt_motion_map(frame_t0, frame_t1, tau=1):
    """
    frame_t0, frame_t1: consecutive grayscale frames (t and t+1)
    Returns: (motion_energy_map, direction_map)
      motion_energy_map -- magnitude of directional motion at each pixel
      direction_map      -- signed horizontal motion (sign = direction)
    """
    f0 = frame_t0.astype(np.float32)
    f1 = frame_t1.astype(np.float32)

    # spatial neighbor (horizontal correlator pair: pixel x and x+1)
    f0_shift = np.roll(f0, -1, axis=1)
    f1_shift = np.roll(f1, -1, axis=1)

    # classic HR correlator: A(t)*B(t-tau) - B(t)*A(t-tau)
    arm1 = f1 * f0_shift
    arm2 = f1_shift * f0
    direction_map = arm1 - arm2

    motion_energy_map = np.abs(direction_map)
    motion_energy_map = cv2.normalize(motion_energy_map, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return motion_energy_map, direction_map


def detect_motion_region(frame_t0_gray, frame_t1_gray, energy_thresh=15):
    """
    Convenience wrapper: given two grayscale frames, returns the bounding
    box of the single largest detected motion region (or None if nothing
    crosses the threshold), plus the raw energy map.
    """
    energy, direction = reichardt_motion_map(frame_t0_gray, frame_t1_gray)
    _, thresh = cv2.threshold(energy, energy_thresh, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, energy, direction
    largest = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    box = cv2.boundingRect(largest)  # (x, y, w, h)
    return box, energy, direction


# ---------------- VALIDATION TEST (run directly to verify correctness) ----------------
if __name__ == "__main__":
    H, W = 300, 500
    rng = np.random.default_rng(42)
    static_bg = (rng.random((H, W)) * 60 + 20).astype(np.uint8)

    frame_t0 = static_bg.copy()
    frame_t1 = static_bg.copy()
    cv2.circle(frame_t0, (150, 150), 20, 255, -1)
    cv2.circle(frame_t1, (180, 150), 20, 255, -1)

    energy, direction = reichardt_motion_map(frame_t0, frame_t1)
    print("Validation: does the detector isolate MOVING dot from STATIC background?")
    print(f"  Mean energy on static background: {energy[20:280, 350:480].mean():.2f}  (expect ~0)")
    print(f"  Mean energy in the dot's motion path: {energy[120:180, 140:190].mean():.2f}  (expect > 0)")
