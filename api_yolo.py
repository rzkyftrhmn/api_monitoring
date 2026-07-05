import cv2
import threading
import time
import os
import shutil
import subprocess

from flask import Flask, jsonify, Response, send_from_directory
from ultralytics import YOLO
import torch


# ============================================================
# KONFIGURASI
# ============================================================

MODEL_PATH = "yolo11n.pt"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE.startswith("cuda")

HOST = "0.0.0.0"
PORT = 5000

# Karena loop kita sleep 0.5 detik, berarti kurang lebih 2 FPS.
# Nanti kalau stabil baru dinaikkan.
HLS_FPS = 8

CCTV_LIST = {
    "jl_ahmad_jazuli": {
        "nama": "Jl. Ahmad Jazuli",
        "url": "https://cctvjss.jogjakota.go.id/kotabaru/ANPR-Jl-Ahmad-Jazuli.stream/chunklist_w827455557.m3u8"
    },
    "simpang_terban_view_timur": {
        "nama": "SIMPANG TERBAN VIEW TIMUR",
        "url": "https://cctvjss.jogjakota.go.id/atcs/ATCS_Simpang_Terban_View_Timur.stream/chunklist_w2020525505.m3u8"
    },
    "simpang_demangan_view_selatan": {
        "nama": "SIMPANG DEMANGAN VIEW SELATAN",
        "url": "https://cctvjss.jogjakota.go.id/atcs/ATCS_Simpang_Demangan_View_Selatan.stream/chunklist_w735060430.m3u8"
    },
    "simpang_kiai_mojo": {
        "nama": "SIMPANG KIAI MOJO",
        "url": "https://cctvjss.jogjakota.go.id/atcs/ATCS_Lampu_Merah_KyaiMojo.stream/chunklist_w818634083.m3u8"
    },
    "simpang_pingit_1": {
        "nama": "SIMPANG PINGIT 1 - JL.TENTARA PELAJAR",
        "url": "https://cctvjss.jogjakota.go.id/atcs/ATCS_Lampu_Merah_Pingit1.stream/chunklist_w2067257340.m3u8"
    },
}

# Test 1 kamera dulu.
# Kalau sudah berhasil, nanti bisa ganti jadi:
# HLS_ENABLED_KEYS = set(CCTV_LIST.keys())
HLS_ENABLED_KEYS = set(CCTV_LIST.keys())

WARNA_KELAS = {
    2: (0, 255, 0),      # mobil
    3: (0, 165, 255),    # motor
    5: (255, 0, 0),      # bus
    7: (0, 0, 255),      # truk
}

LABEL_KELAS = {
    2: "Mobil",
    3: "Motor",
    5: "Bus",
    7: "Truk",
}


# ============================================================
# INISIALISASI
# ============================================================

app = Flask(__name__)

model = YOLO(MODEL_PATH)
model.to(DEVICE)

print(f"[DEVICE] YOLO running on: {DEVICE}")
if DEVICE.startswith("cuda"):
    print(f"[GPU] {torch.cuda.get_device_name(0)}")

lock = threading.Lock()
model_lock = threading.Lock()

traffic_results = {}
latest_frames = {}

is_running = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HLS_DIR = os.path.join(BASE_DIR, "hls_output")
os.makedirs(HLS_DIR, exist_ok=True)

hls_processes = {}


# ============================================================
# HELPER FFmpeg / HLS
# ============================================================

def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg tidak ditemukan. Install FFmpeg dan pastikan command 'ffmpeg -version' bisa jalan."
        )


def clean_hls_folder(key):
    key_dir = os.path.join(HLS_DIR, key)
    os.makedirs(key_dir, exist_ok=True)

    for filename in os.listdir(key_dir):
        if filename.endswith(".m3u8") or filename.endswith(".ts"):
            try:
                os.remove(os.path.join(key_dir, filename))
            except Exception:
                pass


def start_hls_process(key, frame_width, frame_height, fps=HLS_FPS):
    key_dir = os.path.join(HLS_DIR, key)
    os.makedirs(key_dir, exist_ok=True)

    clean_hls_folder(key)

    playlist_path = os.path.join(key_dir, "index.m3u8").replace("\\", "/")
    segment_path = os.path.join(key_dir, "segment_%03d.ts").replace("\\", "/")

    command = [
        "ffmpeg",
        "-y",

        # Input dari Python berupa raw frame OpenCV BGR
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{frame_width}x{frame_height}",
        "-r", str(fps),
        "-i", "-",

        # Encode video H.264
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-g", str(fps * 2),
        "-sc_threshold", "0",

        # Output HLS
        "-f", "hls",
        "-hls_time", "2",
        "-hls_list_size", "5",
        "-hls_flags", "delete_segments+omit_endlist",
        "-hls_segment_filename", segment_path,
        playlist_path,
    ]

    print(f"[HLS] Starting FFmpeg for {key}")
    print(f"[HLS] Playlist: {playlist_path}")

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        bufsize=0
    )

    hls_processes[key] = process
    return process


def write_hls_frame(key, frame):
    if key not in HLS_ENABLED_KEYS:
        return

    if frame is None:
        return

    height, width = frame.shape[:2]

    # H.264 yuv420p lebih aman kalau width/height genap.
    if width % 2 != 0:
        frame = frame[:, :width - 1]
        width -= 1

    if height % 2 != 0:
        frame = frame[:height - 1, :]
        height -= 1

    process = hls_processes.get(key)

    if process is None or process.poll() is not None:
        process = start_hls_process(key, width, height, HLS_FPS)

    try:
        process.stdin.write(frame.tobytes())
    except Exception as e:
        print(f"[HLS] Gagal menulis frame untuk {key}: {e}")

        try:
            process.kill()
        except Exception:
            pass

        hls_processes.pop(key, None)


# ============================================================
# HELPER TRAFFIC
# ============================================================

def tentukan_status(total: int) -> str:
    if total > 18:
        return "Macet Total"
    elif total > 10:
        return "Padat Merayap"
    elif total > 5:
        return "Ramai Lancar"

    return "Lancar"


# ============================================================
# THREAD PROSES CCTV
# ============================================================

def proses_cctv(key: str, info: dict):
    global is_running

    url = info["url"]
    nama = info["nama"]

    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print(f"[-] Gagal membuka stream: {nama}")
        return

    print(f"[+] Thread aktif -> {nama}")

    while is_running:
        ret, frame = cap.read()

        if not ret:
            print(f"[!] Stream putus: {nama}, mencoba reconnect...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(url)
            continue

        # ---------- DETEKSI YOLO ----------
        with model_lock:
            results = model.predict(
                source=frame,
                classes=[2, 3, 5, 7],
                conf=0.18,
                iou=0.45,
                imgsz=640,
                device=DEVICE,
                half=USE_HALF,
                verbose=False
            )

        motor = mobil = bus = truk = 0

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h

            # Reklasifikasi box kecil menjadi motor
            if class_id in [2, 5, 7] and (box_area < 2000 or box_w < 45):
                class_id = 3

            if class_id == 2:
                mobil += 1
            elif class_id == 3:
                motor += 1
            elif class_id == 5:
                bus += 1
            elif class_id == 7:
                truk += 1

            # ---------- GAMBAR BOUNDING BOX ----------
            warna = WARNA_KELAS.get(class_id, (255, 255, 255))
            label = f"{LABEL_KELAS.get(class_id, '?')} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 2)

            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                1
            )

            cv2.rectangle(
                frame,
                (x1, y1 - th - 6),
                (x1 + tw + 4, y1),
                warna,
                -1
            )

            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        total = motor + mobil + bus + truk
        status = tentukan_status(total)
        waktu = time.strftime("%H:%M:%S")

        # ---------- OVERLAY INFO ----------
        overlay_text = f"{nama} | {status} | {total} kendaraan | {waktu}"

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 32),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            overlay_text,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ---------- KIRIM FRAME KE HLS FFmpeg ----------
        # Ini yang bikin hasil YOLO jadi video .m3u8.
        write_hls_frame(key, frame)

        # ---------- ENCODE FRAME KE JPEG UNTUK MJPEG LAMA ----------
        success, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 70]
        )

        if not success:
            time.sleep(0.12)
            continue

        # ---------- SIMPAN HASIL KE MEMORY ----------
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
                    "bus": bus,
                    "truk": truk,
                },
                "waktu_update": waktu,
            }

        time.sleep(0.9)

    cap.release()
    print(f"[-] Thread berhenti: {nama}")


# ============================================================
# GLOBAL HEADER
# ============================================================

@app.after_request
def add_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Cache-Control"] = "no-cache"
    return response


# ============================================================
# ROUTES - JSON API
# ============================================================

@app.route("/api/traffic", methods=["GET"])
def api_semua():
    with lock:
        data = list(traffic_results.values())

    return jsonify({
        "success": True,
        "data": data,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/api/traffic/<key>", methods=["GET"])
def api_satu(key):
    with lock:
        data = traffic_results.get(key)

    if data is None:
        return jsonify({
            "success": False,
            "message": "Jalan tidak ditemukan atau belum ada data"
        }), 404

    return jsonify({
        "success": True,
        "data": data
    })


@app.route("/api/jalan", methods=["GET"])
def api_daftar_jalan():
    data = [
        {
            "key": key,
            "nama": value["nama"]
        }
        for key, value in CCTV_LIST.items()
    ]

    return jsonify({
        "success": True,
        "data": data
    })


@app.route("/api/health", methods=["GET"])
def health():
    with lock:
        aktif = len(traffic_results)

    return jsonify({
        "success": True,
        "message": "API berjalan",
        "jalan_aktif": aktif,
        "total_jalan": len(CCTV_LIST),
        "device": DEVICE,
        "hls_enabled_keys": list(HLS_ENABLED_KEYS),
    })


# ============================================================
# ROUTES - MJPEG STREAM LAMA
# ============================================================

def generate_stream(key: str):
    while True:
        with lock:
            frame_bytes = latest_frames.get(key)

        if frame_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
        else:
            time.sleep(0.8)

        time.sleep(0.04)


@app.route("/api/stream/<key>")
def video_stream(key):
    if key not in CCTV_LIST:
        return jsonify({
            "success": False,
            "message": "Key tidak valid"
        }), 404

    return Response(
        generate_stream(key),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# ROUTES - HLS VIDEO BARU
# ============================================================

@app.route("/api/hls/<key>/status", methods=["GET"])
def hls_status(key):
    if key not in CCTV_LIST:
        return jsonify({
            "success": False,
            "message": "Key tidak valid"
        }), 404

    key_dir = os.path.join(HLS_DIR, key)
    playlist_path = os.path.join(key_dir, "index.m3u8")

    segments = []

    if os.path.exists(key_dir):
        segments = [
            filename for filename in os.listdir(key_dir)
            if filename.endswith(".ts")
        ]

    return jsonify({
        "success": True,
        "key": key,
        "nama": CCTV_LIST[key]["nama"],
        "hls_enabled": key in HLS_ENABLED_KEYS,
        "playlist_exists": os.path.exists(playlist_path),
        "segments_count": len(segments),
        "playlist_url": f"http://localhost:{PORT}/hls/{key}/index.m3u8",
        "folder": key_dir,
    })


@app.route("/hls/<key>/<path:filename>")
def serve_hls(key, filename):
    if key not in CCTV_LIST:
        return jsonify({
            "success": False,
            "message": "Key tidak valid"
        }), 404

    directory = os.path.join(HLS_DIR, key)

    if filename.endswith(".m3u8"):
        mimetype = "application/vnd.apple.mpegurl"
    elif filename.endswith(".ts"):
        mimetype = "video/mp2t"
    else:
        mimetype = "application/octet-stream"

    return send_from_directory(
        directory,
        filename,
        mimetype=mimetype
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    check_ffmpeg()

    print("=" * 55)
    print("  TRAFFIC DETECTION API - YOLO + HLS")
    print("=" * 55)
    print(f"  Model : {MODEL_PATH}")
    print(f"  Device: {DEVICE}")
    print(f"  Server: http://{HOST}:{PORT}")
    print(f"  HLS   : enabled for {list(HLS_ENABLED_KEYS)}")
    print("=" * 55)

    threads = []

    for key, info in CCTV_LIST.items():
        thread = threading.Thread(
            target=proses_cctv,
            args=(key, info),
            daemon=True
        )

        threads.append(thread)
        thread.start()

    print(f"\n[*] {len(threads)} thread CCTV dijalankan")
    print("[*] Endpoint tersedia:")
    print("    GET  /api/health")
    print("    GET  /api/jalan")
    print("    GET  /api/traffic")
    print("    GET  /api/traffic/<key>")
    print("    GET  /api/stream/<key>")
    print("    GET  /api/hls/<key>/status")
    print("    GET  /hls/<key>/index.m3u8")
    print("\n[*] Server starting...\n")

    app.run(
        host=HOST,
        port=PORT,
        threaded=True,
        debug=False,
        use_reloader=False
    )