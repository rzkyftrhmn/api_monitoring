# Python API Monitoring CCTV

Project ini adalah Python Flask API untuk monitoring lalu lintas dari CCTV. Sistem membaca stream CCTV HLS `.m3u8`, menjalankan deteksi kendaraan dengan YOLO, menggambar bounding box pada frame, lalu menyediakan data realtime dan video hasil deteksi.

## Fitur Utama

- Membaca stream CCTV HLS `.m3u8`.
- Deteksi kendaraan dengan Ultralytics YOLO.
- Kelas kendaraan: motor, mobil, bus, dan truk.
- API JSON untuk data traffic realtime.
- Stream MJPEG fallback lewat `GET /api/stream/{key}`.
- Stream HLS video lewat `GET /hls/{key}/index.m3u8`.
- Status HLS lewat `GET /api/hls/{key}/status`.
- Support GPU CUDA jika tersedia.
- Fallback otomatis ke CPU jika CUDA tidak tersedia.

## Dependency

Project ini membutuhkan:

- Python
- pip
- virtual environment
- FFmpeg
- PyTorch
- Ultralytics YOLO
- OpenCV
- Flask

FFmpeg bukan package Python. FFmpeg adalah software sistem yang harus diinstall terpisah dan tersedia lewat command `ffmpeg`. Di project ini FFmpeg dipakai untuk mengubah frame hasil deteksi YOLO menjadi video HLS `.m3u8` dan segment `.ts`.

## Instalasi FFmpeg Windows

Install FFmpeg dengan `winget`:

```powershell
winget install --id Gyan.FFmpeg -e
```

Setelah install, restart terminal lalu cek:

```powershell
ffmpeg -version
```

Kalau command `ffmpeg` tidak dikenali, pastikan folder `bin` FFmpeg sudah masuk ke environment variable `PATH`.

## Instalasi Python

Buat virtual environment dan install dependency Python:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Pastikan file model YOLO yang dipakai oleh aplikasi tersedia, misalnya `yolo11n.pt` sesuai konfigurasi `MODEL_PATH` di `api_yolo.py`.

## CPU dan GPU

Project bisa berjalan di CPU-only, tetapi proses deteksi akan lebih lambat. Jika ingin memakai GPU NVIDIA, install PyTorch versi CUDA yang sesuai dengan versi driver/CUDA di komputer.

Cek apakah PyTorch mendeteksi GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Kode otomatis memakai `cuda:0` jika CUDA tersedia, dan fallback ke `cpu` jika CUDA tidak tersedia.

## Cara Menjalankan

Jalankan server Flask:

```powershell
python api_yolo.py
```

Server berjalan di:

```text
http://localhost:5000
```

## Endpoint API

- `GET /api/health` - cek status API, device, jumlah jalan aktif, dan key HLS.
- `GET /api/jalan` - daftar key dan nama CCTV/jalan.
- `GET /api/traffic` - data traffic realtime untuk semua CCTV.
- `GET /api/traffic/{key}` - data traffic realtime untuk satu CCTV.
- `GET /api/stream/{key}` - stream MJPEG fallback hasil deteksi.
- `GET /api/hls/{key}/status` - status playlist dan segment HLS untuk satu CCTV.
- `GET /hls/{key}/index.m3u8` - playlist HLS video hasil deteksi.

## Testing

Contoh URL untuk test setelah server berjalan:

- `http://localhost:5000/api/health`
- `http://localhost:5000/api/traffic`
- `http://localhost:5000/api/hls/simpang_mirota/status`
- `http://localhost:5000/hls/simpang_mirota/index.m3u8`

Output HLS dianggap berhasil jika `index.m3u8` bisa dibuka dan berisi teks playlist seperti `#EXTM3U`.

## Troubleshooting

- `ffmpeg not found`: install FFmpeg, restart terminal, lalu pastikan `ffmpeg -version` berhasil. Jika masih gagal, tambahkan folder `bin` FFmpeg ke `PATH`.
- HLS loading terus: cek `GET /api/hls/{key}/status`, pastikan `playlist_exists` bernilai `true` dan `segments_count` lebih dari `0`. Tunggu beberapa detik karena segment HLS dibuat bertahap.
- GPU 100%: kurangi jumlah CCTV yang aktif, gunakan model YOLO yang lebih ringan, turunkan FPS HLS, atau jalankan di CPU jika perlu.
- Motor belum banyak terdeteksi: model COCO kadang kurang optimal untuk kondisi kamera lokal. Solusinya gunakan model yang sudah fine-tune untuk motor/kendaraan lokal atau sesuaikan threshold deteksi.
- Kamera belum punya playlist HLS: pastikan URL CCTV `.m3u8` valid dan bisa dibuka. Jika playlist dari sumber CCTV berubah, update URL pada konfigurasi `CCTV_LIST`.
