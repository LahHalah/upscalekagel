# install.py

# Installing dependencies
import os
import requests

# Fungsi untuk menjalankan perintah shell
# Menggunakan os.system() agar skrip bisa berjalan di lingkungan non-notebook
def run_command(command):
    print(f"-> Menjalankan: {command}")
    # os.system akan menjalankan perintah di shell sistem operasi
    os.system(command)

# Periksa apakah repo GitHub masih hidup
# Fungsi ini memastikan URL repo dapat dijangkau sebelum mencoba mengunduh.
def repo_exists(repo_url):
    try:
        # Menggunakan requests.head untuk memeriksa header tanpa mengunduh konten
        response = requests.head(repo_url, allow_redirects=True, timeout=5)
        # Status code 200 (OK) menunjukkan sumber daya dapat dijangkau
        return response.status_code == 200
    except requests.RequestException as e:
        # Menangkap error jaringan atau timeout
        print(f"Error saat memeriksa repo: {e}")
        return False

# --- Konfigurasi Repositori ---
repo_name = "R3gm/SD_diffusers_interactive"
# Menggunakan URL mentah langsung ke file yang dibutuhkan
raw_repo_path = f"https://raw.githubusercontent.com/{repo_name}/main/tool_shed/sd_runes.py"
# --- Akhir Konfigurasi ---

repo_alive = repo_exists(raw_repo_path)

if repo_alive:
    print(f"\n✅ Repo ditemukan. Mengunduh file dari {raw_repo_path}...")
    # Menggunakan 'wget' untuk mengunduh, dipanggil via run_command
    # -q: quiet, tidak terlalu verbose | --show-progress: menampilkan progress bar
    run_command(f"wget -q --show-progress {raw_repo_path}")
else:
    print(f"\n❌ Peringatan: Repo {repo_name} tidak dapat dijangkau di URL. Tidak dapat mengunduh file.")
    print("Silakan periksa kembali URL atau koneksi internet Anda.")

# --- Instalasi Paket Sistem (Asumsi lingkungan Linux/Debian seperti di Colab/Kaggle) ---
print("\n📦 Memasang paket sistem yang diperlukan untuk pengunduhan yang efisien...")
# apt-get update: memperbarui daftar paket
run_command("apt-get update -qq > /dev/null")
# apt-get install: memasang aria2 dan rar. -y: otomatis 'yes', -qq: sangat quiet
run_command("apt-get install -y -qq aria2 rar > /dev/null")

# --- Membersihkan dan Memasang Paket Python ---
# Membersihkan instalasi yang dapat menyebabkan konflik (sering terjadi di lingkungan pre-instalasi seperti Kaggle/Colab)
print("\n🐍 Membersihkan dan memasang paket Python...")
# pip uninstall: menghapus paket yang berpotensi konflik. -y: otomatis 'yes', -q: quiet
run_command("pip uninstall -y -q ydf grpcio-status thinc tensorflow-decision-forests spacy")

# pip install: Memasang paket yang dibutuhkan dengan versi spesifik. -q: quiet
run_command("pip install -q stablepy==0.6.1 transformers==4.49.0 thinc==8.3.4 blis==1.2.1 spacy==3.8.7 dynamicprompts==0.31.0")

print("\n✨ **Instalasi selesai! Lingkungan siap digunakan.** 🚀")
