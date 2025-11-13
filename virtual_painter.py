"""
virtual_painter.py

Main entry point for the AI Virtual Painter project.
Follows the user's requirements from `MyPromptToFollow.md`.

Usage: python virtual_painter.py

Requirements: see `requirements.txt`.
"""
from typing import List, Tuple
import os
import time

import cv2
import numpy as np

from hand_tracking_module import HandDetector


# ----------------- Configuration / Customization -----------------
HEADER_FOLDER = os.path.join(os.path.dirname(__file__), "header")
BRUSH_THICKNESS = 15
ERASER_THICKNESS = 50
DEFAULT_COLOR = (180, 119, 31)  # Default brush color (BGR) - colorblind-friendly blue

# Colorblind-friendly palette (BGR tuples)
# Order: first three are brush colors; the header's last slot is the eraser.
COLOR_PALETTE: List[Tuple[int, int, int]] = [
    (180, 119, 31),  # Blue  - hex #1f77b4 -> BGR (180,119,31)
    (14, 127, 255),  # Orange- hex #ff7f0e -> BGR (14,127,255)
    (44, 160, 44),   # Green - hex #2ca02c -> BGR (44,160,44)
]

# Header images order should correspond to colors + eraser as last
HEADER_FILES = ["brush1.png", "brush2.png", "brush3.png", "eraser.png"]

# ----------------- Helper functions -----------------

def load_header_images(folder: str, files: List[str]):
    headers = []
    for fname in files:
        path = os.path.join(folder, fname)
        if os.path.isfile(path):
            img = cv2.imread(path)
            if img is None:
                # file exists but could not be read -> create an unreadable placeholder
                ph = np.zeros((125, 640, 3), dtype=np.uint8)
                cv2.putText(ph, fname + " (unreadable)", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                headers.append(ph)
            else:
                headers.append(img)
        else:
            # create placeholder header if file not present
            ph = np.zeros((125, 640, 3), dtype=np.uint8)
            cv2.putText(ph, fname + " (missing)", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            headers.append(ph)
    return headers


def resize_headers_to_frame(headers: List[np.ndarray], frame_width: int, header_height: int = 125) -> List[np.ndarray]:
    """Resize header images to the frame width and a fixed header height.

    This ensures selection x-coordinates map predictably to header slots.
    """
    resized = []
    for h in headers:
        try:
            rh = cv2.resize(h, (frame_width, header_height), interpolation=cv2.INTER_AREA)
        except Exception:
            rh = np.zeros((header_height, frame_width, 3), dtype=np.uint8)
            cv2.putText(rh, "header_error", (10, header_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        resized.append(rh)
    return resized


def overlay_header(frame, header_img):
    """Overlay `header_img` at the top of `frame` using slicing.

    Assumes header_img height <= frame height.
    """
    h, w, _ = header_img.shape
    frame[0:h, 0:w] = header_img
    return frame


# ----------------- Main Application -----------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam (VideoCapture returned false). Check camera access and index.")
        return
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector(detectionCon=0.75, trackCon=0.75)


    # Load headers (raw) - we'll resize them to the actual frame width once we have a frame
    raw_headers = load_header_images(HEADER_FOLDER, HEADER_FILES)
    header_index = 0
    current_color = COLOR_PALETTE[0]
    draw_color = current_color
    brush_thickness = BRUSH_THICKNESS

    # Canvas for drawing will be created after reading the first valid frame
    canvas = None

    xp, yp = 0, 0

    pTime = 0

    while True:
        success, img = cap.read()
        if not success or img is None:
            print("Warning: empty frame received from camera. Exiting.")
            break

        img = cv2.flip(img, 1)

        # initialize canvas and headers once when we know frame size
        if canvas is None:
            h_frame, w_frame = img.shape[:2]
            canvas = np.zeros((h_frame, w_frame, 3), dtype=np.uint8)
            headers = resize_headers_to_frame(raw_headers, w_frame, header_height=125)

        # overlay header
        header = headers[header_index]
        img = overlay_header(img, header)

        # detect hands
        img = detector.find_hands(img, draw=False)
        lmList = detector.find_position(img, draw=False)
        fingers = detector.fingers_up()

        # selection mode: index + middle fingers up
        if len(lmList) != 0:
            x1, y1 = lmList[8][1], lmList[8][2]  # index finger tip
            x2, y2 = lmList[12][1], lmList[12][2]  # middle finger tip

            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                # if in header region -> selection
                h_h, h_w, _ = header.shape
                if y1 < h_h:
                    # determine selection based on x position
                    segment = max(1, h_w // len(headers))
                    sel = min(x1 // segment, len(headers) - 1)
                    header_index = int(sel)
                    # update color/eraser
                    if header_index < len(COLOR_PALETTE):
                        draw_color = COLOR_PALETTE[header_index]
                        brush_thickness = BRUSH_THICKNESS
                    else:
                        # eraser selected
                        draw_color = (0, 0, 0)
                        brush_thickness = ERASER_THICKNESS

                    # small visual feedback
                    cv2.rectangle(img, (x1 - 20, y1 - 20), (x1 + 20, y1 + 20), (255, 255, 255), 2)

            # drawing mode: only index finger up
            elif fingers[1] and not fingers[2]:
                x, y = x1, y1
                if xp == 0 and yp == 0:
                    xp, yp = x, y

                cv2.line(img, (xp, yp), (x, y), draw_color, brush_thickness)
                cv2.line(canvas, (xp, yp), (x, y), draw_color, brush_thickness)
                xp, yp = x, y

        # merge canvas and frame with mask to preserve header
        img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
        img_inv = cv2.bitwise_not(img_inv)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, canvas)

        # draw current tool and FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        tool_text = "Eraser" if draw_color == (0, 0, 0) else "Brush"
        color_text = "Black" if draw_color == (0, 0, 0) else str(draw_color)
        cv2.putText(img, f"Mode: {tool_text} | Color: {color_text}", (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"FPS: {int(fps)}", (1150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("AI Virtual Painter", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # clear canvas (preserve dynamic size)
            if canvas is not None:
                canvas = np.zeros_like(canvas)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()