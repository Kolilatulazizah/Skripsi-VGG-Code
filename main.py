from flask import Flask, request, render_template
from pymongo import MongoClient
import cv2
from PIL import Image
import numpy as np
import base64
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from mtcnn import MTCNN
import time
import os
import datetime
from telebot import *
import threading
import json
from io import BytesIO


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
app = Flask(__name__)

client = MongoClient("<MONGODB_URL>")
api = "<TELEGRAM_API>"
bot = TeleBot(api)
id_admin = "<ADMIN_ID>"


detector = MTCNN()

model = load_model('HasilTrainingvgg_kelas40_10.h5', compile=False)
Label_Nama = json.load(open("model_kelas40_10.json"))


def telegram_polling_thread():
    @bot.message_handler(commands=["ADDID"])
    def tambahid(message):

        db = client['datapresensi']
        collection = db['telegram_id_data']
        data = message.text.replace("/ADDID", "").strip()

        if data != "":
            query = {"Nama": data}
            existing_doc = collection.find_one(query)
            if existing_doc:
                bot.send_message(message.chat.id, f"ID anda sudah tersimpan dalam database, berikut ID telegram anda\nID: {message.chat.id}\n")
            else:
                data = {
                    "TeleID": str(message.chat.id),
                    "Nama": data
                }
                # Insert the new document into the collection
                collection.insert_one(data)
                bot.send_message(message.chat.id, f"ID: {message.chat.id}\nID anda telah tersimpan")
        else:
            bot.send_message(message.chat.id, f"Mohon gunakan format berikut\n\n /ADDID <nama_anda>")

    @bot.message_handler(commands=["rekapdata"])
    def rekapdata(message):

        db = client['datapresensi']
        collection1 = db['telegram_id_data']
        collection2 = db['log']

        query1 = {"TeleID": str(message.chat.id)}
        telegram_doc = collection1.find_one(query1)
        if telegram_doc:
            bulan = time.strftime('%m')
            tahun = time.strftime('%Y')
            query2 = {"Nama":telegram_doc["Nama"], "Tahun":tahun, "Bulan":bulan}
            presensi_log = collection2.find(query2)
            if presensi_log:
                total_waktu_data = 0
                for data in presensi_log:
                    total_waktu = data.get("Total_waktu")
                    hours, minutes, seconds = map(int, total_waktu.split(':'))
                    total_waktu_data += (hours * 3600) + (minutes * 60) + seconds
                bot.send_message(message.chat.id, f'Rekap data {telegram_doc["Nama"]} di bulan {bulan} tahun {tahun} \nAnda telah masuk selama {total_waktu_data//3600} jam pada bulan ini')
            else:
                bot.send_message(message.chat.id, f'Rekap data {telegram_doc["Nama"]} di bulan {bulan} tahun {tahun} \nAnda telah masuk selama 0 jam pada bulan ini')
    bot.polling()







@app.route("/")
def awal():
    return render_template("index.html")

@app.route("/proses", methods=["POST"])
def proses():
    gambar = request.files['image']
    img = Image.open(gambar.stream)

    img = img.convert("RGB")
    img = np.asarray(img)
    umt_img = cv2.UMat(img)
    if len(detector.detect_faces(img)) > 0:
        data = detector.detect_faces(img)[0]['box']
        x = data[0]
        y = data[1]
        w = data[2]
        h = data[3]
        wajah = img[y:y+h,x:x+w]
        cv2.rectangle(umt_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        wajah = cv2.resize(wajah,(224, 224))
        if np.sum([wajah])!=0:
            wajah = (wajah.astype('float')/127.5)-1
            wajah = img_to_array(wajah)
            wajah = np.expand_dims(wajah,axis=0)
            prediksi_nama = model.predict(wajah)[0]
            threshold_nama = str(round(max(prediksi_nama), 3))
            if max(prediksi_nama) > 0.1:
                label=Label_Nama[prediksi_nama.argmax()]
                cv2.putText(umt_img, label, (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 2,  (0, 255, 0), 2, cv2.LINE_AA, False)
            else:
                return {"msg":"Threshold_Rendah"}

        img = umt_img.get()
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, encoded_img = cv2.imencode('.jpg', image)
        base64_img = base64.b64encode(encoded_img).decode('utf-8')
        data_uri = 'data:image/jpeg;base64,' + base64_img

        db = client['datapresensi']
        collection = db['nama_nim_batch3']
        label_data = collection.find_one({"Nama":label})
        if label_data is None:
            label_data = {}
        nim = label_data.get("NIM", "-")
        email = label_data.get("email", "-")
        study_program = label_data.get("study_program", "-")
        batch_year = label_data.get("batch_year", "-")
        project = label_data.get("project", "-")
        waktu = time.strftime("%H:%M:%S")
        tanggal=time.strftime("%d/%m/%Y")
        return {
                "msg":"Success",
                "data":{
                    "img":data_uri,
                    "waktu":waktu,
                    "tanggal":tanggal,
                    "nama":label,
                    "nim":nim,
                    "email":email,
                    "study_program":study_program,
                    "batch_year":batch_year,
                    "project":project,
                    "threshold":threshold_nama
                    }
                }
    return {"msg":"Failed"}




@app.route('/hasil', methods=["POST"])
def hasil():
    nim = request.form["nim"].strip()
    gambar = request.form["Gambar"]
    kondisi = request.form["Kondisi"]
    nilai_threshold = request.form["Threshold"]
    nama_lama = request.form["Nama"]
    tanggal = request.form["Tanggal"]
    waktu = request.form["Waktu"]

    db = client['datapresensi']
    collection = db['nama_nim_batch3']
    label_data = collection.find_one({"NIM":nim})
    if label_data is None:
        label_data = {}
    data_nama = label_data.get('Nama', '-')
    _, base64_string = gambar.split(',')
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    if kondisi == "Benar":
        image.save(f"./Hasil_data/Benar/{data_nama}_{nilai_threshold}.jpg")
    else:
        image.save(f"./Hasil_data/Salah/{nama_lama}_{data_nama}_{nilai_threshold}.jpg")


    db1 = client['datapresensi']
    collection1 = db1['log']

    tanggal_db, bulan_db, tahun_db = tanggal.split("/")

    query = {"Nama": data_nama, "Tanggal": tanggal_db, "Bulan": bulan_db, "Tahun": tahun_db}
    existing_doc1 = collection1.find_one(query)
    if existing_doc1:
        new_waktu_list = existing_doc1.get("Waktu")
        new_waktu_list.append(waktu)
        start = datetime.strptime(new_waktu_list[0], "%H:%M:%S")
        end = datetime.strptime(new_waktu_list[-1], "%H:%M:%S")
        total_waktu = str(end-start)
        collection1.update_one(query, {"$set": {"Waktu": new_waktu_list, 'Total_waktu': total_waktu}})
    else:
        data = {
            "Nama": data_nama,
            "Tanggal": tanggal_db,
            "Bulan": bulan_db,
            "Tahun": tahun_db,
            "Waktu": [waktu],  # Create a new list containing only the new waktu value
            "Total_waktu": "0"
        }
        # Insert the new document into the collection
        collection1.insert_one(data)
    collection2 = db1["telegram_id_data"]
    query1 = {"Nama": data_nama}
    existing_doc2 = collection2.find_one(query1)
    existing_doc1 = collection1.find_one(query)
    bot.send_photo(id_admin, photo=image_data, caption=f"Notifikasi Admin\n{tanggal} -- {waktu}\n\nPresensi atas nama {data_nama}\n\nModel Kelas 40_2")
    if existing_doc2:
        if len(existing_doc1.get("Waktu")) == 1:
            bot.send_photo(existing_doc2.get("TeleID"), photo=image_data, caption=f"{tanggal} -- {waktu}\n\nPresensi kedatangan atas nama {data_nama}")
        else:
            bot.send_photo(existing_doc2.get("TeleID"), photo=image_data, caption=f"{tanggal} -- {waktu}\n\nPresensi pulang atas nama {data_nama}\nLama waktu: {existing_doc1.get('Total_waktu')}")
    return {"msg":"Success"}


telegram_thread = threading.Thread(target=telegram_polling_thread)
telegram_thread.daemon = True  # Set the thread as a daemon so it exits when the main program exits
telegram_thread.start()

app.run(host="0.0.0.0", port=3019, ssl_context='adhoc')
#app.run()
