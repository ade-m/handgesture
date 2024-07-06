import cv2
import mediapipe as mp
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np

# Inisialisasi MediaPipe Hands.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi pipeline untuk klasifikasi gambar.
processor = AutoImageProcessor.from_pretrained("RavenOnur/Sign-Language")
model = AutoModelForImageClassification.from_pretrained("RavenOnur/Sign-Language")
pipe = pipeline("image-classification", model=model, feature_extractor=processor)

# Daftar untuk menyimpan riwayat label.
label_history = []

def classify_hand_gesture(image):
    # Konversi gambar OpenCV (numpy array) menjadi gambar PIL.
    pil_image = Image.fromarray(image)
    # Gunakan pipeline untuk klasifikasi.
    results = pipe(pil_image)
    return results

def draw_label_history(image, label_history):
    # Tampilkan riwayat label di bagian bawah frame.
    h, w, _ = image.shape
    for i, label in enumerate(label_history[-5:]):  # Menampilkan 5 label terakhir
        y = h - (i + 1) * 30
        cv2.putText(image, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def main():
    cap = cv2.VideoCapture(1)  # Gunakan kamera dengan indeks 1.
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Konversi frame ke format RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Proses deteksi tangan.
        results = hands.process(image)

        # Gambar landmark pada frame asli.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Potong gambar tangan dari frame.
                h, w, c = image.shape
                xmin = ymin = 1
                xmax = ymax = 0
                for lm in hand_landmarks.landmark:
                    xmin = min(xmin, lm.x)
                    ymin = min(ymin, lm.y)
                    xmax = max(xmax, lm.x)
                    ymax = max(ymax, lm.y)
                xmin = int(xmin * w)
                ymin = int(ymin * h)
                xmax = int(xmax * w)
                ymax = int(ymax * h)

                # Tambahkan margin pada bounding box.
                margin = 20  # Sesuaikan margin sesuai kebutuhan.
                xmin = max(xmin - margin, 0)
                ymin = max(ymin - margin, 0)
                xmax = min(xmax + margin, w)
                ymax = min(ymax + margin, h)
                
                hand_img = image[ymin:ymax, xmin:xmax]

                # Gambar kotak di sekitar tangan yang terdeteksi.
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

                if hand_img.size > 0:
                    # Klasifikasikan gambar tangan.
                    classification_results = classify_hand_gesture(hand_img)

                    # Tampilkan hasil klasifikasi di bawah tangan.
                    if classification_results:
                        label = classification_results[0]['label']
                        confidence = classification_results[0]['score']
                        text = f'{label}: {confidence:.2f}'
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                        text_x = xmin
                        text_y = ymax + text_size[1] + 10
                        if text_y + text_size[1] > h:
                            text_y = ymin - 10
                        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Tambahkan label ke riwayat.
                        label_history.append(text)

        # Tampilkan riwayat label di bagian bawah frame.
        draw_label_history(image, label_history)

        # Tampilkan frame dengan deteksi tangan dan hasil klasifikasi.
        cv2.imshow('Hand Detection and Classification', image)

        # Periksa apakah tombol 'q' ditekan untuk keluar.
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
