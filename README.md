# 🛡️ Secure Image Upscaler & Face Restorer (StablePy-AES)

Aplikasi Gradio ini menyediakan solusi peningkatan gambar (upscaling) dan restorasi wajah (face restoration) yang didukung oleh pustaka `StablePy` dan dilengkapi dengan lapisan keamanan tingkat lanjut. **Semua output gambar disimpan di disk hanya dalam format terenkripsi (AES-256 GCM) dan hanya dapat diakses melalui dekripsi *on-the-fly* di memori.**

## ✨ Fitur Utama

  * **Peningkatan Kualitas Gambar (Upscaling):** Mendukung berbagai model canggih seperti **RealESRNet**, **4x-UltraSharp**, **HAT-L\_SRx4**, dan lainnya, menggunakan pustaka `StablePy`.
  * **Restorasi Wajah Opsional:** Integrasi model **GFPGAN**, **CodeFormer**, dan **RestoreFormer** untuk memperbaiki wajah yang buram atau rusak dalam gambar.
  * **Keamanan Data Tinggi (AES-256 GCM):** Gambar output tidak pernah disimpan sebagai file terbuka di disk. Mereka segera dienkripsi menggunakan kunci yang berasal dari kata sandi pengguna melalui **PBKDF2** (untuk keamanan kunci).
  * **Dekripsi Memori-Saja:** Gambar yang didekripsi untuk pengunduhan diproses langsung di memori dan ditransmisikan ke browser melalui **Data URI Base64**, memastikan file yang tidak terenkripsi tidak pernah menyentuh disk host (seperti lingkungan Kaggle).
  * **Tiling dan Half-Precision Support:** Dukungan *tiling* untuk pemrosesan gambar beresolusi sangat tinggi dan opsi *half-precision* (FP16) untuk GPU NVIDIA (CUDA) untuk efisiensi memori dan kecepatan.

## 🚀 Instalasi dan Prasyarat

Aplikasi ini memerlukan Python 3.8+ dan beberapa pustaka utama.

### 1\. Klon Repositori

```bash
git clone https://github.com/LahHalah/upscalekagel
cd upscalekagel
```

### 2\. Instalasi Dependensi

Kami menggunakan `pycryptodomex` untuk kriptografi yang kuat dan `stablepy` untuk pemrosesan gambar.

```bash
!python install.py
```

> **Catatan:** Jika Anda menjalankan skrip langsung (seperti di lingkungan Kaggle), skrip akan secara otomatis memeriksa dan menginstal `pycryptodomex` jika diperlukan.

## 💡 Cara Menggunakan

### 1\. Jalankan Aplikasi Gradio

```bash
!python app.py 
```

### 2\. Proses Gambar (Enkripsi)

1.  **Unggah Gambar Input.**
2.  **Tentukan Kata Sandi Enkripsi (Wajib):** Ini adalah kunci untuk enkripsi/dekripsi. **Jangan Lupakan Kata Sandi Ini.**
3.  Sesuaikan parameter Upscaler (Model, Skala, Tiling).
4.  Sesuaikan parameter Restorasi Wajah (opsional).
5.  Klik **`1. 🚀 Proses Gambar & Enkripsi`**.

Setelah selesai, gambar hasil upscale yang **terenkripsi** akan disimpan di disk di folder `/kaggle/working/output` (atau `/output` tergantung lingkungan). Data terenkripsi juga disimpan dalam status memori Gradio.

### 3\. Unduh Gambar Terdekripsi (Dekripsi Memori-Saja)

1.  **Masukkan kembali Kata Sandi Enkripsi yang sama** (untuk validasi dan dekripsi).
2.  Klik **`2. ⬇️ Unduh Gambar Terdekripsi`**.

Aplikasi akan mendekripsi data dari memori dan menghasilkan tautan unduhan HTML yang memungkinkan Anda mengunduh gambar beresolusi penuh langsung ke komputer Anda.

## 🛠️ Pengembangan Lanjutan

**Kontribusi dipersilakan\!** Silakan ajukan *pull request* untuk model upscaler baru, fitur restorasi wajah, atau perbaikan keamanan.
