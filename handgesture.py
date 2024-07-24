import cv2
import mediapipe as mp

# Fungsi untuk mendeteksi kamera yang tersedia
def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

# Inisialisasi Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Fungsi untuk mengenali gestur tangan
def recognize_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Thumbs Up
    if (thumb_tip.y < index_finger_tip.y and
        thumb_tip.y < middle_finger_tip.y and
        thumb_tip.y < ring_finger_tip.y and
        thumb_tip.y < pinky_tip.y):
        return "Thumbs Up"

    # Peace Sign (Index and Middle finger up)
    if (index_finger_tip.y < thumb_tip.y and
        middle_finger_tip.y < thumb_tip.y and
        ring_finger_tip.y > thumb_tip.y and
        pinky_tip.y > thumb_tip.y):
        return "Peace Sign"

    # Fist
    if (thumb_tip.y > index_finger_tip.y and
        thumb_tip.y > middle_finger_tip.y and
        thumb_tip.y > ring_finger_tip.y and
        thumb_tip.y > pinky_tip.y):
        return "Fist"

    # Metal Sign (Index and Pinky finger up)
    if (index_finger_tip.y < thumb_tip.y and
        pinky_tip.y < thumb_tip.y and
        middle_finger_tip.y > thumb_tip.y and
        ring_finger_tip.y > thumb_tip.y):
        return "Metal"


    return "Unknown Gesture"

# Fungsi untuk mendeteksi dan mengenali gerakan tangan
def detect_hand_gesture(image, hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = recognize_gesture(hand_landmarks)

            # Mendapatkan posisi terendah dari tangan
            h, w, c = image.shape
            max_y = 0
            for landmark in hand_landmarks.landmark:
                max_y = max(max_y, int(landmark.y * h))

            # Mendapatkan posisi horizontal tengah tangan
            # Misalnya, untuk teks yang ditempatkan di tengah tangan
            center_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)

            # Menempatkan teks di bawah tangan
            cv2.putText(image, gesture, (center_x - 50, max_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return image

# Mendeteksi kamera yang tersedia
available_cameras = list_cameras()
print(f"Kamera yang tersedia: {available_cameras}")

# Memilih kamera
selected_camera_index = int(input(f"Pilih kamera (indeks) {available_cameras}: "))

# Memilih kamera
selected_camera_index = int(input(f"Pilih kamera (indeks) {available_cameras}: "))

# Membuka kamera yang dipilih
cap = cv2.VideoCapture(selected_camera_index)

if not cap.isOpened():
    print("Tidak dapat membuka kamera")
    exit()

# Loop untuk menangkap frame dari kamera dan mendeteksi gerakan tangan
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Gagal menangkap frame")
        break

    frame = detect_hand_gesture(frame, hands)
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan sumber daya kamera dan menutup jendela
cap.release()
cv2.destroyAllWindows()
