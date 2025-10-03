# Installing dependencies
import os
import requests
from IPython.display import clear_output

# Periksa apakah repo GitHub masih hidup
# Fungsi ini memastikan URL repo dapat dijangkau sebelum mencoba mengunduh.
def repo_exists(repo_url):
    try:
        response = requests.head(repo_url, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

repo_name = "R3gm/SD_diffusers_interactive"
# Menggunakan URL mentah langsung ke file yang dibutuhkan
raw_repo_path = f"https://raw.githubusercontent.com/{repo_name}/main/tool_shed/sd_runes.py"
repo_alive = repo_exists(raw_repo_path)

if repo_alive:
    print(f"Mengunduh file dari {raw_repo_path}...")
    !wget -q --show-progress {raw_repo_path}
else:
    print(f"Peringatan: Repo {repo_name} tidak dapat dijangkau. Tidak dapat mengunduh file.")

# Memasang paket sistem yang diperlukan untuk pengunduhan yang efisien
print("Memasang paket sistem...")
!apt-get update -qq > /dev/null
!apt-get install -y -qq aria2 rar > /dev/null

# Membersihkan dan memasang paket Python
# Membersihkan instalasi yang dapat menyebabkan konflik di Kaggle
print("Membersihkan dan memasang paket Python...")
!pip uninstall -y -q ydf grpcio-status thinc tensorflow-decision-forests spacy
!pip install -q stablepy==0.6.1 transformers==4.49.0 thinc==8.3.4 blis==1.2.1 spacy==3.8.7 dynamicprompts==0.31.0

print("\nInstalasi selesai! 🚀")
# Membersihkan output sel untuk menjaga notebook tetap rapi
clear_output()
print("Lingkungan siap.")
