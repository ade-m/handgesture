import cv2
import mediapipe as mp

#insialisasi mediapipe hand
mp_hands = mp.solutions.hands
hands= mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

#fungsi untuk mengenali gesture tangan
def recognize_gesture(hand_landmarks):
    #ambil posisi ujung jari
    ujung_jempol = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    ujung_telunjuk = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    ujung_jariTengah = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ujung_jarimanis = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ujung_kelingking = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    #thumbs up => jika hanya jempol yang diangkat
    if(ujung_jempol.y < ujung_telunjuk.y and
       ujung_jempol.y < ujung_jariTengah.y and
       ujung_jempol.y < ujung_jarimanis.y and
       ujung_jempol.y < ujung_kelingking.y):
        return "Thumbs Up"
    
    # Peace Sign (Index and Middle finger up)
    if (ujung_telunjuk.y < ujung_jempol.y and
        ujung_jariTengah.y < ujung_jempol.y and
        ujung_jarimanis.y > ujung_jempol.y and
        ujung_kelingking.y > ujung_jempol.y):
        return "Peace Sign"

    # Fist
    if (ujung_jempol.y > ujung_telunjuk.y and
        ujung_jempol.y > ujung_jariTengah.y and
        ujung_jempol.y > ujung_jarimanis.y and
        ujung_jempol.y > ujung_kelingking.y):
        return "Fist"

    # Metal Sign (Index and Pinky finger up)
    if (ujung_telunjuk.y < ujung_jempol.y and
        ujung_kelingking.y < ujung_jempol.y and
        ujung_jariTengah.y > ujung_jempol.y and
        ujung_jarimanis.y > ujung_jempol.y):
        return "Metal"
    
    return "Gesture tidak diketahui"

#fungsi untuk mendeteksi tangan dengan mediapipe
def detect_hand_gesture(image,hand):
    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results = hand.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #deteksi gesture
            gesture = recognize_gesture(hand_landmarks)
            #menggambar tangan terdeteksi pada frame/image
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            #menampilkan teks gesture
            cv2.putText(image,gesture,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)

    return image

#membuka kamera 1
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("tidak dapat membuka kamera")
    exit()

while(cap.isOpened):
    ret,frame = cap.read()
    if not ret:
        print("Gagal menangkap frame")
        break

    frame = detect_hand_gesture(frame,hands)
    #tampilkan frame
    cv2.imshow("Hand gesture Recognition dengan Mediapipe dan OpenCV",frame)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()