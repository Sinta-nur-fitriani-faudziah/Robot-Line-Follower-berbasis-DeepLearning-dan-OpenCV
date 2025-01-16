import cv2
import os

# Fungsi untuk membuat direktori untuk menyimpan dataset
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Fungsi untuk menangkap gambar dari webcam dan menyimpannya ke dalam dataset
def capture_images(parent_directory, object_name, num_samples, camera_index):
    directory = os.path.join(parent_directory, object_name)
    create_directory(directory)
    cap = cv2.VideoCapture(camera_index)
    count = 0

    while True:
        ret, frame = cap.read()
        cv2.imshow('Capturing Images', frame)

        # Menyimpan gambar ke dalam dataset
        if cv2.waitKey(1) & 0xFF == ord('s') and count < num_samples:
            img_name = os.path.join(directory, f"{object_name}_{count}.png")
            cv2.imwrite(img_name, frame)
            print(f"Captured {object_name}_{count}.png")
            count += 1

        # Keluar dari loop setelah jumlah sampel mencapai target
        if count >= num_samples:
            break

        # Keluar dari loop jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Input dari pengguna untuk nama objek, jumlah sampel, dan indeks webcam
object_name = input("Masukkan nama objek: ")
num_samples = int(input("Masukkan jumlah sampel yang ingin Anda ambil: "))
camera_index = int(input("Masukkan indeks webcam USB (biasanya indeks lebih tinggi dari webcam bawaan): "))

# Memanggil fungsi capture_images untuk menangkap sampel
capture_images("dataset", object_name, num_samples, camera_index)
