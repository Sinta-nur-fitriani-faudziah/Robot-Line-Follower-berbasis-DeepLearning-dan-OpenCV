import cv2
import numpy as np
import serial
from keras.models import load_model
import time

# Hubungkan ke Arduino
arduino = serial.Serial('COM14', 9600)  # Sesuaikan port serial Arduino

# Muat model machine learning
model = load_model(r"C:\Users\darni\Documents\KP Bolabot\Line Follower Anfis\best_model.keras")

# Label dan tindakan
label_to_action = {0: "F", 1: "L", 2:"R"}

# Variabel untuk menyimpan perintah sebelumnya
last_action = None


# Fungsi untuk mengirimkan aksi ke Arduino
def send_to_arduino(action):
    global last_action
    if action != last_action:  # Hanya kirim jika perintah berbeda dari sebelumnya
        arduino.write(f"{action}\n".encode())
        last_action = action  # Perbarui perintah terakhir
        time.sleep(0.1)  # Tunggu sejenak agar Arduino bisa memproses

# Fungsi utama untuk klasifikasi dan kontrol
def classify_and_control():
    cap = cv2.VideoCapture(1)

    # Konfigurasi kamera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Tidak dapat membaca frame.")
                break

            # Preprocessing gambar untuk model
            resized_frame = cv2.resize(frame, (100, 100))
            normalized_frame = resized_frame / 255.0
            input_data = np.expand_dims(normalized_frame, axis=0)

            # Prediksi menggunakan model
            predictions = model.predict(input_data, verbose=0)
            predicted_class = np.argmax(predictions)

            # Ambil tindakan berdasarkan prediksi
            action = label_to_action[predicted_class]
            send_to_arduino(action)  # Kirim ke Arduino

            # Tampilkan hasil pada layar
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Action: {action}", (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Robot Control", frame)

            # Tekan 'q' untuk keluar
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Bersihkan sumber daya
        cap.release()
        cv2.destroyAllWindows()
        print("Program selesai.")

# Panggil fungsi utama
classify_and_control()
