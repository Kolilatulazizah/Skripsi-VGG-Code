from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import time
import math
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from mtcnn import MTCNN
import json
import pyttsx3
import os
import threading
import base64
from telebot import *
import threading
import datetime


api = "<TELEGRAM_BOT_API>"
bot = TeleBot(api)

def telegram_polling_thread():
    @bot.message_handler(commands=["cekID"])
    def awal(message):
        bot.send_message(message.chat.id, f"ID: {message.chat.id}")
    bot.polling()


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



gambar = mp.solutions.drawing_utils
tangan = mp.solutions.hands



detector = MTCNN()
model = load_model(r'HasilTrainingvgg_Epoch_04.h5', compile=False)
Label_Nama = json.load(open(r"model_kelas50.json"))


end_presensi = 0
start_presensi = 0
elapsed_time = 0
presensi = []
hasil_gambar = None

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def proses_presensi(image, pointx, pointy):
    data_point = {}
    check_point = []
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(len(detector.detect_faces(img)))
    if len(detector.detect_faces(img)) > 0:
        for idx, face in enumerate(detector.detect_faces(img)):
            data = face["box"]
            x = data[0]
            y = data[1]
            w = data[2]
            h = data[3]
            middle_pointx, middle_pointy = int(x+(w//2)), int(y+(h//2))
            jarak = calculate_distance((pointx, pointy), (middle_pointx, middle_pointy))
            data_point[idx] = [x, y, w, h]
            check_point.append(jarak)

        min_index = check_point.index(min(check_point))
        x, y, w, h = data_point[min_index]
        wajah = img[y:y+h,x:x+w]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        wajah = cv2.resize(wajah,(224, 224))
        if np.sum([wajah])!=0:
            wajah = (wajah.astype('float')/127.5) - 1
            wajah = img_to_array(wajah)
            wajah = np.expand_dims(wajah,axis=0)
            prediksi_nama = model.predict(wajah)[0]
            label=Label_Nama[prediksi_nama.argmax()]
            cv2.putText(img, label, (x, y-2),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,150,71), 4)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, encoded_img = cv2.imencode('.png', img)
            base64_img = base64.b64encode(encoded_img).decode('utf-8')
            data_uri = 'data:image/jpeg;base64,' + base64_img
            return img, label, data_uri, str(round(max(prediksi_nama) * 100, 3))
        return None



# Function to speak asynchronously
def speak_async(text):
    engine = pyttsx3.init()
    engine.setProperty('voice', r'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_idID_Andika')
    engine.say(text)
    engine.runAndWait()
    engine.stop()


app = Flask(__name__)

def presensi_wajah():
    global hasil_gambar
    label = ""
    try:
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        with tangan.Hands(model_complexity=0,
                          max_num_hands=1,
                          static_image_mode=False,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as hands:

            while cap.isOpened():
                count = 0
                marklist = []
                _, img = cap.read()
                img = cv2.flip(img, 1)
                image1 = img.copy()
                image = img.copy()
                if img is None:
                    break

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img.flags.writeable = False
                hasil = hands.process(img)

                if hasil.multi_handedness:
                    label = hasil.multi_handedness[0].classification[0].label
                    if label == "Left":
                        label = "Right"
                    elif label == "Right":
                        label = "Left"

                if hasil.multi_hand_landmarks:
                    for num, hand in enumerate(hasil.multi_hand_landmarks):
                        gambar.draw_landmarks(image, hand, tangan.HAND_CONNECTIONS)
                    bBox = cv2.boundingRect(np.array([[landmark.x * img.shape[1], landmark.y * img.shape[0]]
                                                     for landmark in hand.landmark]).astype(np.int32))
                    center_x = int(bBox[0] + int(bBox[2]//2))
                    center_y = int(bBox[1] + int(bBox[3]//2))

                    hand = hasil.multi_hand_landmarks[0]
                    for id, landMark in enumerate(hand.landmark):
                        imgH, imgW, imgC = img.shape
                        xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
                        marklist.append([id, xPos, yPos, label])

                if len(marklist) != 0:
                    if marklist[4][3] == "Right" and marklist[4][1] > marklist[3][1]:  # Right Thumb
                        count = count + 1
                    elif marklist[4][3] == "Left" and marklist[4][1] < marklist[3][1]:  # Left Thumb
                        count = count + 1
                    if marklist[8][2] < marklist[6][2]:  # Index finger
                        count = count + 1
                    if marklist[12][2] < marklist[10][2]:  # Middle finger
                        count = count + 1
                    if marklist[16][2] < marklist[14][2]:  # Ring finger
                        count = count + 1
                    if marklist[20][2] < marklist[18][2]:  # Little finger
                        count = count + 1

                img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                cv2.putText(image, f"Count: {count}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if count == 5:
                    elapsed_time = time.time() - start_time
                    lama_detik = 2
                    if elapsed_time < lama_detik:
                        loading_percentage = int((elapsed_time / lama_detik) * 100)
                        cv2.putText(image, f"{loading_percentage}%", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4)

                        radius = 50
                        loading_angle = int((elapsed_time / lama_detik) * 360)
                        cv2.ellipse(image, (center_x+30, center_y-10), (radius, radius), 0, 0, loading_angle, (0, 255, 0), 6)

                    elif elapsed_time >= lama_detik:
                        elapsed_time = 0
                        hasil_gambar = proses_presensi(img, marklist[9][1], marklist[9][2])
                        if hasil_gambar is not None:
                            if float(hasil_gambar[3]) >= 10:
                                if not hasil_gambar[1] in presensi:
                                    presensi.append(hasil_gambar[1])
                                    start_presensi = time.time()
                                    threading.Thread(target=speak_async, args=(f"Selamat datang {hasil_gambar[1]}",)).start()
                                    waktu = time.strftime("%H:%M:%S")
                                    tanggal=time.strftime("%d/%m/%Y")
                                    label = hasil_gambar[1]
                                    skor = hasil_gambar[3]
                                    gambar_bbox = hasil_gambar[0]
                                    _, encoded_img = cv2.imencode('.jpg', gambar_bbox)
                                    frame = encoded_img.tobytes()
                                    bot.send_photo("<ID_ADMIN_1>", photo=frame, caption=f"{tanggal} -- {waktu}\n\nPresensi atas nama {label}\nSimilarity: {skor}")
                                    bot.send_photo("<ID_ADMIN_2>", photo=frame, caption=f"{tanggal} -- {waktu}\n\nPresensi atas nama {label}\nSimilarity: {skor}")

                                    cv2.imwrite(f"Hasil/{tanggal.replace('/', '')}_{waktu.replace(':', '')}.png", image1)
                                    cv2.imwrite(f"Hasil bbox/{tanggal.replace('/', '')}_{waktu.replace(':', '')}.png", gambar_bbox)

                                else:
                                    start_presensi = time.time()
                                    threading.Thread(target=speak_async, args=(f"{hasil_gambar[1]} telah melakukan presensi",)).start()
                                    waktu = time.strftime("%H:%M:%S")
                                    tanggal=time.strftime("%d/%m/%Y")
                                    label = hasil_gambar[1]
                                    skor = hasil_gambar[3]
                                    gambar_bbox = hasil_gambar[0]
                                    _, encoded_img = cv2.imencode('.jpg', gambar_bbox)
                                    frame = encoded_img.tobytes()
                                    bot.send_photo("<ID_ADMIN_1>", photo=frame, caption=f"{tanggal} -- {waktu}\n\n telah melakukan presensi {label}\nSimilarity: {skor}")
                                    bot.send_photo("<ID_ADMIN_2>", photo=frame, caption=f"{tanggal} -- {waktu}\n\n telah melakukan presensi {label}\nSimilarity: {skor}")

                                    cv2.imwrite(f"Hasil/{tanggal.replace('/', '')}_{waktu.replace(':', '')}.png", image1)
                                    cv2.imwrite(f"Hasil bbox/{tanggal.replace('/', '')}_{waktu.replace(':', '')}.png", gambar_bbox)
                            else:
                                #pass
                                threading.Thread(target=speak_async, args=(f"kemiripan anda berada di bawah 90 persen, silahkan ulangi",)).start()


                else:
                    start_time = time.time()
                end_presensi = time.time()
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        print(e)


@app.route("/")
def awal():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(presensi_wajah(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/json_data')
def data_json():
    global hasil_gambar
    if hasil_gambar is not None:  # Assuming hasil_gambar is the variable containing the data you want to jsonify
        data1, data2, data3 = hasil_gambar[1], hasil_gambar[2], hasil_gambar[3]
        hasil_gambar = None
        return jsonify(
            {
                "message": "Available",
                "Name": data1,
                "Img": data2,
                "Similarity": data3
            })
    else:
        return jsonify({"message": "NoData",})
telegram_thread = threading.Thread(target=telegram_polling_thread)
telegram_thread.daemon = True  # Set the thread as a daemon so it exits when the main program exits
telegram_thread.start()

app.run()
