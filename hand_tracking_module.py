"""
hand_tracking_module.py

OOP wrapper around MediaPipe Hands for easy reuse.
Provides:
- HandDetector.find_hands(img, draw=True)
- HandDetector.find_position(img, handNo=0, draw=True)
- HandDetector.fingers_up() -> List[bool]

Requires: mediapipe, opencv-python, numpy
"""
from typing import List, Tuple, Optional

import cv2
import mediapipe as mp


class HandDetector:
    """Hand detector using MediaPipe Hands.

    Methods
    -------
    find_hands(img, draw=True):
        Detects hands and draws landmarks on `img` if requested.

    find_position(img, handNo=0, draw=True):
        Returns a list of landmark tuples [(id, x, y), ...] for the chosen hand.

    fingers_up():
        Returns a list of 5 booleans indicating whether each finger is up (thumb..pinky).
    """

    def __init__(
        self,
        mode: bool = False,
        maxHands: int = 1,
        detectionCon: float = 0.7,
        trackCon: float = 0.7,
    ) -> None:
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

        self.results = None
        self.lmList: List[Tuple[int, int, int]] = []
        self.handType: str = "Right"

    def find_hands(self, img, draw: bool = True) -> object:
        """Detect hands in `img` and optionally draw landmarks.

        Returns the processed `img` (BGR) to allow chaining.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, handNo: int = 0, draw: bool = True) -> List[Tuple[int, int, int]]:
        """Populate and return `self.lmList` for the requested hand number.

        Each list item is a tuple: (id, x, y) with pixel coordinates.
        """
        self.lmList = []

        if not self.results or not self.results.multi_hand_landmarks:
            return self.lmList

        if handNo >= len(self.results.multi_hand_landmarks):
            return self.lmList

        myHand = self.results.multi_hand_landmarks[handNo]
        h, w, c = img.shape
        for id, lm in enumerate(myHand.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            self.lmList.append((id, cx, cy))
            if draw:
                cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

        # determine hand type (Left / Right) if available
        if self.results.multi_handedness and len(self.results.multi_handedness) > handNo:
            self.handType = self.results.multi_handedness[handNo].classification[0].label

        return self.lmList

    def fingers_up(self) -> List[bool]:
        """Return list of 5 booleans for fingers [thumb, index, middle, ring, pinky].

        Uses landmark positions and the detected `handType` to heuristically
        detect whether a finger is up.
        """
        fingers = [False, False, False, False, False]
        if not self.lmList:
            return fingers

        # Thumb: compare tip and IP in x direction (hand-dependent)
        try:
            # self.tipIds[0] == 4, compare x with landmark 3
            if self.handType == "Right":
                fingers[0] = self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]
            else:
                fingers[0] = self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]
        except Exception:
            fingers[0] = False

        # Other fingers: tip y < pip y -> finger is up
        for i in range(1, 5):
            try:
                fingers[i] = self.lmList[self.tipIds[i]][2] < self.lmList[self.tipIds[i] - 2][2]
            except Exception:
                fingers[i] = False

        return fingers


if __name__ == "__main__":
    # quick local test harness (optional)
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.find_hands(img)
        lm = detector.find_position(img)
        fingers = detector.fingers_up()
        cv2.putText(img, f"Fingers: {fingers}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("HandDetector Test", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
