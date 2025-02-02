import cv2
from mtcnn import MTCNN
import os

try:
    os.mkdir("dataset wajah")
except:
    pass


detector = MTCNN()

path = r"C:\semester 7\data skripsi\dataset presensi"
for i in os.listdir(path):
    nama_folder = i.replace("-samples", "")
    try:
        os.mkdir(f"dataset wajah\\{nama_folder}")
    except:
        pass
    for gambar in os.listdir(f"{path}\\{i}"):
        if gambar not in os.listdir(f"dataset wajah\\{nama_folder}"):
            print(f"dataset wajah\\{nama_folder}\\{gambar}")
            try:
                img = cv2.imread(f'{path}\\{i}\\{gambar}')
                pic = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                data = detector.detect_faces(pic)[0]['box']
                x = data[0]
                y = data[1]
                w = data[2]
                h = data[3]
                wajah = img[y:y+h,x:x+w]
                cv2.imwrite(f"dataset wajah\\{nama_folder}\\{gambar}", wajah)
            except:
                print("error")
        print()
