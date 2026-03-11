# Virtual AI drawing tool
# Author: Bekarys Anuarbek


import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
canvas = None
prev_pos = {
    "Left": (0, 0),
    "Right": (0, 0)
}
color = (0, 255, 0)
brush_size = 5
mode = 'DRAW'
prev_time = 0
history = []
last_undo_time = 0
undo_delay = 0.8


cv2.namedWindow('Virtual Drawing', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Virtual Drawing', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if canvas is None:
        canvas = np.zeros_like(frame)
    
    frame = cv2.flip(frame, 1)
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
    prev_time = current_time
    
    cv2.rectangle(frame, (0, 0), (1280, 90), (50, 50, 50), -1) 
    cv2.rectangle(frame, (20, 20), (120, 70), (0, 0, 255), -1)
    cv2.rectangle(frame, (140, 20), (240, 70), (0, 255, 0), -1)
    cv2.rectangle(frame, (260, 20), (360, 70), (255, 0, 0), -1)
    cv2.rectangle(frame, (380, 20), (480, 70), (0, 255, 255), -1)

    cv2.rectangle(frame, (520, 20), (650, 70), (0, 0, 0), -1)
    cv2.putText(frame, 'Clear', (535, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

  
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            hand_label = results.multi_handedness[idx].classification[0].label
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

            index_up = index_tip.y < index_pip.y
            middle_up = middle_tip.y < middle_pip.y
            thumb_up = thumb_tip.y < thumb_ip.y
            pinky_up = pinky_tip.y < pinky_pip.y and not index_up and not middle_up and not thumb_up

            distance = abs(index_tip.x - thumb_tip.x)

            if distance < 0.03:
                brush_size = 5
            elif distance < 0.06:
                brush_size = 10
            else:
                brush_size = 20

            x = int(index_tip.x * frame.shape[1])
            y = int(index_tip.y * frame.shape[0])
            y_middle = int(middle_tip.y * frame.shape[0])
            prev_x, prev_y = prev_pos[hand_label]

            cv2.circle(frame, (x, y), 10, (255, 255, 255), -1)

            if y < 90:
                if 20 < x < 120:
                    color = (0, 0, 255)
                elif 140 < x < 240:
                    color = (0, 255, 0)
                elif 260 < x < 360:
                    color = (255, 0, 0)
                elif 380 < x < 480:
                    color = (0, 255, 255)
                elif 520 < x < 650:
                    canvas = np.zeros_like(frame)

            if index_up and not middle_up:

                mode = 'DRAW'

                if prev_x == 0 and prev_y == 0:
                    history.append(canvas.copy())

                if prev_x == 0 and prev_y == 0:
                    prev_pos[hand_label] = (x, y)
                    continue

                x = int(prev_x *0.5 + x*0.5)
                y = int(prev_y *0.5 + y*0.5)

                cv2.line(canvas, (prev_x, prev_y), (x, y), color, brush_size)
                prev_pos[hand_label] = (x, y)
            
            elif index_up and middle_up:

                mode = 'ERASE'
                
                cv2.circle(canvas, (x, y), 40, (0, 0, 0), -1)
                prev_pos[hand_label] = (0, 0)

            elif thumb_up and not index_up and not middle_up:

                cv2.imwrite("drawing.png", canvas)
                mode = 'SAVE'

            elif pinky_up and len(history) > 0 and time.time() - last_undo_time > undo_delay:

                canvas = history.pop()
                mode = 'UNDO'
                last_undo_time = time.time()

            else:
                prev_pos[hand_label] = (0, 0)
            
    frame = cv2.add(frame, canvas)
    
    cv2.rectangle(frame, (1100, 20), (1200, 70), color, -1)
    cv2.putText(frame, 'Color', (1100, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Mode: {mode}', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Brush Size: {brush_size}', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('Virtual Drawing', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('s'):

        screenshot = np.zeros_like(frame)
        screenshot = cv2.add(screenshot, canvas)
        cv2.putText(screenshot, f'Mode:{mode}', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(screenshot, f'Brush Size: {brush_size}', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imwrite("drawing_screenshot.png", screenshot)

cap.release()
cv2.destroyAllWindows()