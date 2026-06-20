import cv2
import threading
import time
import ssl
from ultralytics import YOLO
from flask import Flask, jsonify, Response
import numpy as np

# KONFIGURASI
MODEL_PATH = "yolo11s.pt"
HOST = "0.0.0.0"
PORT = 5000

CCTV_LIST = {
    "jl_pasopati": {
        "nama": "Jl. Pasopati",
        "url": "https://pelindung.bandung.go.id:3443/video/DAHUA/PASTEUR.m3u8"
    },
    "jl_cibaduyut": {
        "nama": "Jl. Cibaduyut",
        "url": "https://pelindung.bandung.go.id:3443/video/HIKSVISION/Cibaytps.m3u8"
    },
    "jl_pasteur": {
        "nama": "Jl. Pasteur",
        "url": "https://pelindung.bandung.go.id:3443/video/HIKSVISION/DrDjunjunanBTC.m3u8"
    },
    "jl_ahmad_yani": {
        "nama": "Jl. Ahmad Yani",
        "url": "https://pelindung.bandung.go.id:3443/video/HIKSVISION/AhmadYanipertigaanmalabardua.m3u8"
    },
    "surapati_gasibu": {
        "nama": "Surapati - Gasibu",
        "url": "https://pelindung.bandung.go.id:3443/video/DAHUA/Surat.m3u8"
    },
}

# Warna bounding box per kelas
WARNA_KELAS = {
    2: (0, 255, 0),    # mobil  -> hijau
    3: (0, 165, 255),  # motor  -> orange
    5: (255, 0, 0),    # bus    -> biru
    7: (0, 0, 255),    # truk   -> merah
}

LABEL_KELAS = {
    2: "Mobil",
    3: "Motor",
    5: "Bus",
    7: "Truk",
}

# INISIALISASI
app   = Flask(__name__)
model = YOLO(MODEL_PATH)
lock  = threading.Lock()

# Simpan hasil deteksi (data JSON) dan frame terakhir tiap jalan
traffic_results: dict = {}
latest_frames:   dict = {}   # key -> JPEG bytes (sudah di-encode)
is_running = True


# HELPER: tentukan status kemacetan
def tentukan_status(total: int) -> str:
    if total > 18:
        return "Macet Total"
    elif total > 10:
        return "Padat Merayap"
    elif total > 5:
        return "Ramai Lancar"
    return "Lancar"


# THREAD: proses 1 CCTV
def proses_cctv(key: str, info: dict):
    global is_running

    url  = info["url"]
    nama = info["nama"]

    # OpenCV butuh ini agar tidak validasi SSL ketat untuk stream HTTPS
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print(f"[-] Gagal membuka stream: {nama}")
        return

    print(f"[+] Thread aktif -> {nama}")

    while is_running:
        ret, frame = cap.read()

        # Jika stream putus, coba reconnect
        if not ret:
            print(f"[!] Stream putus: {nama}, mencoba reconnect...")
            cap.release()
            time.sleep(3)
            cap = cv2.VideoCapture(url)
            continue

        # ---------- DETEKSI YOLO ----------
        results = model(
            frame,
            classes=[2, 3, 5, 7],
            conf=0.18,
            iou=0.45,
            verbose=False
        )

        motor = mobil = bus = truk = 0

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            conf     = float(box.conf[0])

            # Reklasifikasi box kecil -> motor
            box_w    = x2 - x1
            box_h    = y2 - y1
            box_area = box_w * box_h
            if class_id in [2, 5, 7] and (box_area < 2000 or box_w < 45):
                class_id = 3

            # Hitung per kelas
            if class_id == 2:   mobil += 1
            elif class_id == 3: motor += 1
            elif class_id == 5: bus   += 1
            elif class_id == 7: truk  += 1

            # ---------- GAMBAR BOUNDING BOX ----------
            warna = WARNA_KELAS.get(class_id, (255, 255, 255))
            label = f"{LABEL_KELAS.get(class_id, '?')} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 2)
            # Background label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), warna, -1)
            cv2.putText(
                frame, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 1, cv2.LINE_AA
            )

        total  = motor + mobil + bus + truk
        status = tentukan_status(total)
        waktu  = time.strftime("%H:%M:%S")

        # ---------- OVERLAY INFO DI FRAME ----------
        overlay_text = f"{nama} | {status} | {total} kendaraan | {waktu}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(
            frame, overlay_text,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2, cv2.LINE_AA
        )

        # ---------- ENCODE FRAME KE JPEG ----------
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

        # ---------- SIMPAN KE VARIABEL GLOBAL (thread-safe) ----------
        with lock:
            latest_frames[key] = jpeg.tobytes()
            traffic_results[key] = {
                "key": key,
                "nama": nama,
                "status": status,
                "total_kendaraan": total,
                "detail": {
                    "motor": motor,
                    "mobil": mobil,
                    "bus":   bus,
                    "truk":  truk,
                },
                "waktu_update": waktu,
            }

        # Jeda agar CPU tidak terlalu panas saat proses 5 stream
        time.sleep(0.5)

    cap.release()
    print(f"[-] Thread berhenti: {nama}")


# ============================================================
# FLASK ROUTES
# ============================================================

# --- Status semua jalan (JSON) ---
@app.route("/api/traffic", methods=["GET"])
def api_semua():
    with lock:
        data = list(traffic_results.values())
    return jsonify({
        "success": True,
        "data": data,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


# --- Status 1 jalan (JSON) ---
@app.route("/api/traffic/<key>", methods=["GET"])
def api_satu(key):
    with lock:
        data = traffic_results.get(key)
    if data is None:
        return jsonify({"success": False, "message": "Jalan tidak ditemukan atau belum ada data"}), 404
    return jsonify({"success": True, "data": data})


# --- Stream MJPEG 1 jalan (video dengan bounding box) ---
def generate_stream(key: str):
    while True:
        with lock:
            frame_bytes = latest_frames.get(key)

        if frame_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes +
                b"\r\n"
            )
        else:
            # Kirim frame kosong/placeholder jika belum ada data
            time.sleep(0.5)

        time.sleep(0.1)  # 10 fps di stream


@app.route("/api/stream/<key>")
def video_stream(key):
    if key not in CCTV_LIST:
        return jsonify({"success": False, "message": "Key tidak valid"}), 404
    return Response(
        generate_stream(key),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# --- Daftar semua key & nama jalan ---
@app.route("/api/jalan", methods=["GET"])
def api_daftar_jalan():
    data = [{"key": k, "nama": v["nama"]} for k, v in CCTV_LIST.items()]
    return jsonify({"success": True, "data": data})


# --- Health check ---
@app.route("/api/health", methods=["GET"])
def health():
    with lock:
        aktif = len(traffic_results)
    return jsonify({
        "success": True,
        "message": "API berjalan",
        "jalan_aktif": aktif,
        "total_jalan": len(CCTV_LIST),
    })


# MAIN: jalankan semua thread lalu start Flask
if __name__ == "__main__":
    print("=" * 55)
    print("  TRAFFIC DETECTION API - BANDUNG")
    print("=" * 55)
    print(f"  Model : {MODEL_PATH}")
    print(f"  Server: http://{HOST}:{PORT}")
    print("=" * 55)

    # Jalankan thread per CCTV
    threads = []
    for key, info in CCTV_LIST.items():
        t = threading.Thread(target=proses_cctv, args=(key, info), daemon=True)
        threads.append(t)
        t.start()

    print(f"\n[*] {len(threads)} thread CCTV dijalankan")
    print("[*] Endpoint tersedia:")
    print(f"    GET  /api/health              -> cek status server")
    print(f"    GET  /api/jalan               -> daftar semua jalan")
    print(f"    GET  /api/traffic             -> data status semua jalan")
    print(f"    GET  /api/traffic/<key>       -> data status 1 jalan")
    print(f"    GET  /api/stream/<key>        -> video live + bounding box")
    print(f"\n  Contoh key: jl_pasopati, jl_cibaduyut, jl_pasteur,")
    print(f"              jl_ahmad_yani, surapati_gasibu")
    print("\n[*] Server starting...\n")

    # Jalankan Flask (threaded=True agar bisa handle banyak request bersamaan)
    app.run(host=HOST, port=PORT, threaded=True)