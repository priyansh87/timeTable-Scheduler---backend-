import pickle
import numpy as np
import cv2
import face_recognition


# --- SMILE DETECTION HELPER ---
def get_mar(top_lip, bottom_lip):
    """Calculates Mouth Aspect Ratio"""
    A = np.linalg.norm(np.array(top_lip[2]) - np.array(bottom_lip[10]))
    B = np.linalg.norm(np.array(top_lip[3]) - np.array(bottom_lip[9]))
    C = np.linalg.norm(np.array(top_lip[4]) - np.array(bottom_lip[8]))
    D = np.linalg.norm(np.array(top_lip[0]) - np.array(top_lip[6]))
    if D == 0: return 0
    return (A + B + C) / (2.0 * D)


# --- SETUP ---
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

try:
    imgBackground = cv2.imread('resources/blackbackground.png')
    if imgBackground is None: raise FileNotFoundError
except:
    imgBackground = np.zeros((720, 1280, 3), dtype=np.uint8)

print("Loading Encoded file...")
try:
    file = open('EncodeFile.p', 'rb')
    encodingListwithIds = pickle.load(file)
    encodingList, studentIdList = encodingListwithIds
    file.close()
    print("Encode file loaded")
except:
    print("Error: EncodeFile.p not found. Run encodegenerator.py first.")
    exit()

# --- DYNAMIC LIVENESS CONSTANTS ---
SMILE_THRESH = 0.45
NEUTRAL_THRESH = 0.32
COUNTER = 0
LIVENESS_CONFIRMED = False
STATE = "WAITING_FOR_NEUTRAL"

while True:
    success, img = cap.read()
    if not success: break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurrFrame = face_recognition.face_locations(imgS)
    encodingCurrFrame = face_recognition.face_encodings(imgS, faceCurrFrame)

    try:
        imgBackground[162:162 + 480, 55:55 + 640] = img
    except:
        imgBackground[0:480, 0:640] = img

        # --- LIVENESS LOGIC ---
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locs_scaled = []
    for (y1, x2, y2, x1) in faceCurrFrame:
        face_locs_scaled.append((y1 * 4, x2 * 4, y2 * 4, x1 * 4))

    face_landmarks_list = face_recognition.face_landmarks(img_rgb, face_locs_scaled)

    if len(face_landmarks_list) > 0:
        face_landmarks = face_landmarks_list[0]
        top_lip = face_landmarks['top_lip']
        bottom_lip = face_landmarks['bottom_lip']
        mar = get_mar(top_lip, bottom_lip)

        # STATE MACHINE
        if STATE == "WAITING_FOR_NEUTRAL":
            if mar < NEUTRAL_THRESH:
                STATE = "READY_FOR_SMILE"
            status_text = "Step 1: Neutral Face"
            status_color = (0, 255, 255)  # Yellow

        elif STATE == "READY_FOR_SMILE":
            if mar > SMILE_THRESH:
                COUNTER += 1
                if COUNTER >= 5:
                    LIVENESS_CONFIRMED = True
                    STATE = "VERIFIED"
            else:
                COUNTER = 0
            status_text = "Step 2: SMILE NOW!"
            status_color = (0, 0, 255)  # Red

        elif STATE == "VERIFIED":
            status_text = "LIVENESS CONFIRMED"
            status_color = (0, 255, 0)  # Green

        # Overlay Status
        cv2.putText(imgBackground, f"MAR: {mar:.2f} | {status_text}", (55, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    else:
        # Reset if face lost
        COUNTER = 0
        LIVENESS_CONFIRMED = False
        STATE = "WAITING_FOR_NEUTRAL"

    # --- RECOGNITION LOGIC ---
    for encodeFace, FaceLoc in zip(encodingCurrFrame, faceCurrFrame):
        matches = face_recognition.compare_faces(encodingList, encodeFace)
        face_dist = face_recognition.face_distance(encodingList, encodeFace)
        matchIndex = np.argmin(face_dist)

        if matches[matchIndex] and face_dist[matchIndex] < 0.5:
            y1, x2, y2, x1 = FaceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            color = (0, 255, 0) if LIVENESS_CONFIRMED else (0, 0, 255)
            cv2.rectangle(imgBackground, (x1 + 55, y1 + 162), (x2 + 55, y2 + 162), color, 2)

            if LIVENESS_CONFIRMED:
                cv2.putText(imgBackground, str(studentIdList[matchIndex]), (x1 + 55, y1 + 162 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Reset after match
                # LIVENESS_CONFIRMED = False
                # STATE = "WAITING_FOR_NEUTRAL"
            else:
                cv2.putText(imgBackground, "SPOOF CHECK", (x1 + 55, y1 + 162 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Attendance System', imgBackground)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break