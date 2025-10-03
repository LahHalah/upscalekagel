# =========================================================================
# Skrip Utama - Peningkat Gambar dan Restorasi Wajah
# Perubahan: Mengembalikan semua model upscaler yang dihapus.
#            Default upscaler_half_checkbox diubah ke False.
# =========================================================================

# --- Bagian 1: Impor Pustaka yang Diperlukan ---
import gradio as gr
import os
import sys
import subprocess
import datetime
import hashlib
import base64
import requests
import torch
from io import BytesIO
from PIL import Image
from typing import Optional

# Pustaka Eksternal yang Kuat untuk Kriptografi
try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
except ImportError:
    pass # Akan ditangani di bagian __main__

# Pustaka StablePy
try:
    from stablepy import BUILTIN_UPSCALERS, load_upscaler_model
    from stablepy.face_restoration.main_face_restoration import (
        load_face_restoration_model,
        process_face_restoration,
    )
except ImportError:
    print("Peringatan: Pustaka 'stablepy' mungkin hilang. Instalasi diperlukan.")
    pass

# Pustaka untuk kompatibilitas otomatis webui (dijaga)
try:
    from modules.api import api
    from modules.images import save_image
except ImportError:
    api = None
    save_image = None
    pass

# Tentukan direktori output untuk menyimpan file terenkripsi
KAGGLE_OUTPUT_DIR = "/kaggle/working/output"

# --- Bagian 2: Fungsi Enkripsi dan Dekripsi yang Kuat ---

def generate_key_from_password(password: str, salt: bytes, iterations: int = 100000) -> bytes:
    """
    Menghasilkan kunci enkripsi 32-byte dari kata sandi dan salt menggunakan PBKDF2.
    """
    kdf = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations)
    return kdf

def encrypt_image(image: Image.Image, password: str, format: str = "JPEG", quality: int = 90) -> bytes:
    """
    Mengubah gambar menjadi biner (JPEG untuk kecepatan), mengenkripsinya,
    dan mengembalikan data terenkripsi (salt + nonce + tag + ciphertext).
    """
    if not password:
        raise ValueError("Password tidak boleh kosong untuk enkripsi.")
    
    # 1. Konversi Gambar ke Biner (Image -> Bytes)
    img_buffer = BytesIO()
    image.save(img_buffer, format=format, quality=quality)
    image_data = img_buffer.getvalue()
    
    # 2. Enkripsi (AES-256 GCM)
    salt = get_random_bytes(16)
    nonce = get_random_bytes(12) # GCM Nonce = 12 bytes
    
    key = generate_key_from_password(password, salt)
    
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    
    ciphertext, tag = cipher.encrypt_and_digest(image_data)
    
    # 3. Gabungkan semua bagian untuk output terenkripsi yang aman
    encrypted_data = salt + nonce + tag + ciphertext
    
    return encrypted_data

def decrypt_image(encrypted_data: bytes, password: str) -> Image.Image:
    """
    Mendekripsi data biner terenkripsi dan mengembalikannya sebagai objek gambar.
    """
    if not password:
        raise ValueError("Password tidak boleh kosong untuk dekripsi.")
    
    # 1. Ekstrak komponen dari data terenkripsi
    if len(encrypted_data) < 44: # 16 (salt) + 12 (nonce) + 16 (tag)
        raise ValueError("Data terenkripsi rusak atau tidak lengkap.")

    salt = encrypted_data[:16]
    nonce = encrypted_data[16:28]
    tag = encrypted_data[28:44]
    ciphertext = encrypted_data[44:]
    
    # 2. Dekripsi
    key = generate_key_from_password(password, salt)
    
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    
    try:
        decrypted_data = cipher.decrypt_and_verify(ciphertext, tag)
    except ValueError as e:
        raise ValueError("Password salah atau file terenkripsi rusak.") from e
    
    # 3. Konversi Biner ke Gambar (Bytes -> Image)
    img_buffer = BytesIO(decrypted_data)
    try:
        image = Image.open(img_buffer)
        image.load() 
    except Exception as e:
        raise ValueError(f"Gagal memuat gambar dari data terdekripsi: {e}")
    
    return image

# --- Bagian 3: Fungsi Skrip Gradio yang Diperbarui ---

# Perangkat untuk pemrosesan
cl_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Memeriksa perangkat: {'GPU' if cl_device == 'cuda' else 'CPU'} ditemukan.")

def get_calculated_dimensions(original_width: int, original_height: int, scale_factor: float, min_dim: int = 64) -> tuple[int, int]:
    """Menghitung dimensi baru setelah downscale, memastikan dimensi minimal."""
    if scale_factor >= 1.0:
        return original_width, original_height
    
    if scale_factor <= 0.0:
        return min_dim, min_dim

    new_width_proportional = int(original_width * scale_factor)
    new_height_proportional = int(original_height * scale_factor)

    if new_width_proportional < min_dim or new_height_proportional < min_dim:
        if new_width_proportional < new_height_proportional:
            scale_ratio = min_dim / max(1, new_width_proportional)
        else:
            scale_ratio = min_dim / max(1, new_height_proportional)
            
        new_width = int(original_width * scale_factor * scale_ratio)
        new_height = int(original_height * scale_factor * scale_ratio)
        
        new_width = max(min_dim, new_width)
        new_height = max(min_dim, new_height)
    else:
        new_width = new_width_proportional
        new_height = new_height_proportional
    
    return new_width, new_height

def downscale_image_by_factor(image: Image.Image, scale_factor: float, min_dim: int = 64, progress=gr.Progress()) -> Image.Image:
    """Melakukan downscaling gambar dengan LANCZOS."""
    original_width, original_height = image.size
    
    progress(0, desc="Menghitung dimensi baru...")

    if scale_factor >= 1.0:
        progress(1, desc=f"Faktor skala {scale_factor:.2f} (>=1.0), tidak ada downscaling dilakukan.")
        return image

    calculated_width, calculated_height = get_calculated_dimensions(original_width, original_height, scale_factor, min_dim)

    print(f"Gambar asli: {original_width}x{original_height}, Skala input: {scale_factor:.2f}, Downscaling ke: {calculated_width}x{calculated_height}")
    downscaled_img = image.resize((calculated_width, calculated_height), Image.LANCZOS)
    progress(1, desc="Downscaling selesai.")
    return downscaled_img

# KAMUS MODEL UPSCALER DIKEMBALIKAN KE VERSI ASLI
UPSCALER_DICT_GUI = {
    **{bu: bu for bu in BUILTIN_UPSCALERS if bu not in ["None", None]},
    "4xNomosWebPhoto_RealPLKSR": "https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_esrgan/4xNomosWebPhoto_esrgan.pth",
    "4xNomosWebPhoto_esrgan": "https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_esrgan/4xNomosWebPhoto_esrgan.pth",
    "4xRealWebPhoto_v4_dat2": "https://github.com/Phhofm/models/releases/download/4xRealWebPhoto_v4_dat2/4xRealWebPhoto_v4_dat2.pth",
    "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    "4x-UltraSharp": "https://huggingface.co/Shandypur/ESRGAN-4x-UltraSharp/resolve/main/4x-UltraSharp.pth",
    "8x_NMKD-Superscale_150000_G": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/8x_NMKD-Superscale_150000_G.pth",
    "4x_NMKD-Siax_200k": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Siax_200k.pth",
    "4x_NMKD-Superscale-SP_178000_G": "https://huggingface.co/gemasai/4x_NMKD-Superscale-SP_178000_G/resolve/main/4x_NMKD-Superscale-SP_178000_G.pth",
    "4x_foolhardy_Remacri": "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth",
    "1xDeJPG_realplksr_otf": "https://github.com/Phhofm/models/releases/download/1xDeJPG_realplksr_otf/1xDeJPG_realplksr_otf.pth",
    "4xPurePhoto-RealPLSKR.pth": "https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/4xPurePhoto-RealPLSKR.pth",
    "Remacri4xExtraSmoother": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/Remacri%204x%20ExtraSmoother.pth",
    "AnimeSharp4x": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/AnimeSharp%204x.pth",
    "lollypop": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/lollypop.pth",
    "RealisticRescaler4x": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/RealisticRescaler%204x.pth",
    "NickelbackFS4x": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/NickelbackFS%204x.pth",
    "Valar4x": "https://huggingface.co/halffried/gyre_upscalers/resolve/main/esrgan_valar_x4/4x_Valar_v1.pth",
    "HAT_GAN_SRx4": "https://huggingface.co/halffried/gyre_upscalers/resolve/main/hat_ganx4/Real_HAT_GAN_SRx4.safetensors",
    "HAT-L_SRx4": "https://huggingface.co/halffried/gyre_upscalers/resolve/main/hat_lx4/HAT-L_SRx4_ImageNet-pretrain.safetensors",
    "Ghibli_Grain": "https://huggingface.co/anonderpling/upscalers/resolve/main/ESRGAN/ghibli_grain.pth",
    "Detoon4x": "https://huggingface.co/anonderpling/upscalers/resolve/main/ESRGAN/4x_detoon_225k.pth",
    "4xRealWebPhoto_v4_dat2_dup": "https://github.com/Phhofm/models/releases/download/4xRealWebPhoto_v4_dat2/4xRealWebPhoto_v4_dat2.pth", # Duplikat dipertahankan dengan nama unik
}

# Direktori model lokal
directory_upscalers = os.path.join(os.getcwd(), 'upscalers_models')
os.makedirs(directory_upscalers, exist_ok=True)

def download_model(url: str, output_dir: str, progress=gr.Progress()) -> str:
    """Mengunduh file dari URL dengan progress bar."""
    filename = os.path.basename(url)
    output_path = os.path.join(output_dir, filename)
    
    if os.path.exists(output_path):
        progress(1, desc=f"File sudah ada: {filename}")
        print(f"File sudah ada: {output_path}")
        return output_path
    
    progress(0, desc=f"Mengunduh {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded_size += len(chunk)
                if total_size:
                    progress_val = downloaded_size / total_size
                    if int(progress_val * 100) % 5 == 0:
                        progress(progress_val, desc=f"Mengunduh {filename}...")
                        
        progress(1, desc=f"Unduhan selesai: {filename}")
        print(f"Unduhan selesai: {output_path}")
        
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"Gagal mengunduh model dari {url}: {e}") from e
        
    return output_path

# Penyimpanan model global (dijaga)
global_face_restoration_model = None
global_upscaler_model_instance = None
global_current_upscaler_path = None
global_current_face_restorer_name = None
global_current_upscaler_config = {}

# Perangkat untuk pemrosesan (diinisialisasi ulang di Bagian 3)
cl_device = "cuda" if torch.cuda.is_available() else "cpu"

def load_models_cached(face_restoration_type, upscaler_name_key, upscaler_tile, upscaler_tile_overlap, upscaler_half, progress=gr.Progress()):
    """Memuat model dengan caching dan penanganan unduhan."""
    global global_face_restoration_model, global_upscaler_model_instance, \
             global_current_upscaler_path, global_current_face_restorer_name, \
             global_current_upscaler_config

    # --- 1. Memuat Model Restorasi Wajah ---
    if face_restoration_type != "Disabled":
        if global_face_restoration_model is None or global_current_face_restorer_name != face_restoration_type:
            progress(0.1, desc=f"Memuat model restorasi wajah: {face_restoration_type}")
            try:
                global_face_restoration_model = load_face_restoration_model(face_restoration_type, cl_device)
                global_current_face_restorer_name = face_restoration_type
                progress(0.3, desc=f"Model restorasi wajah {face_restoration_type} dimuat.")
            except Exception as e:
                raise gr.Error(f"Gagal memuat model restorasi wajah {face_restoration_type}: {e}") from e
        else:
            progress(0.3, desc=f"Model restorasi wajah {face_restoration_type} sudah dimuat (cache).")
    else:
        global_face_restoration_model = None
        global_current_face_restorer_name = None
        progress(0.3, desc="Restorasi wajah dinonaktifkan.")

    # --- 2. Memuat Model Upscaler ---
    
    target_upscaler_path = UPSCALER_DICT_GUI.get(upscaler_name_key)
    if target_upscaler_path is None:
        raise gr.Error(f"Model upscaler '{upscaler_name_key}' tidak ditemukan dalam kamus.")
        
    if "https://" in str(target_upscaler_path):
        progress(0.4, desc="Mengunduh/Memverifikasi model upscaler...")
        target_upscaler_path = download_model(target_upscaler_path, directory_upscalers, progress=progress)
        progress(0.6, desc=f"Model upscaler {os.path.basename(target_upscaler_path)} tersedia.")

    # Catatan: Kita tetap memasukkan upscaler_half dalam konfigurasi untuk caching
    current_upscaler_params_for_load = {
        'model': target_upscaler_path,
        'device': cl_device,
        'tile': upscaler_tile,
        'tile_overlap': upscaler_tile_overlap,
        'half': upscaler_half # Gunakan nilai dari input (default False)
    }

    if (global_upscaler_model_instance is None or
        global_current_upscaler_path != target_upscaler_path or
        global_current_upscaler_config != current_upscaler_params_for_load):

        progress(0.7, desc=f"Memuat model upscaler: {upscaler_name_key}...")
        try:
            global_upscaler_model_instance = load_upscaler_model(**current_upscaler_params_for_load)
            global_current_upscaler_path = target_upscaler_path
            global_current_upscaler_config = current_upscaler_params_for_load
            progress(1, desc=f"Model upscaler {upscaler_name_key} dimuat.")
        except Exception as e:
            raise gr.Error(f"Gagal memuat model upscaler {upscaler_name_key}: {e}") from e
    else:
        progress(1, desc=f"Model upscaler {upscaler_name_key} sudah dimuat (cache).")

    return global_face_restoration_model, global_upscaler_model_instance

# Fungsi utama untuk memproses dan mengenkripsi gambar
def process_and_encrypt(
    input_image: Optional[Image.Image],
    downscale_factor_value: float,
    model_upscaler_key: str,
    scale_of_the_image_x: float,
    upscaler_tile: int,
    upscaler_tile_overlap: int,
    face_restoration_type: str,
    face_restoration_visibility: float,
    face_restoration_weight: float,
    upscaler_half: bool,
    encryption_password: str,
    progress=gr.Progress(track_tqdm=True)
):
    if not encryption_password:
        raise gr.Error("Mohon masukkan kata sandi di kotak teks.")
    if input_image is None:
        raise gr.Error("Mohon unggah gambar.")

    if cl_device == "cpu" and upscaler_half:
        # Jika pengguna mencoba menggunakan Half-Precision di CPU, nonaktifkan dan beri peringatan
        upscaler_half = False
        gr.Warning("Upscaler Presisi Setengah dinonaktifkan untuk CPU.")
    
    # 1. Muat/Cek Model
    try:
        print("Memuat atau memeriksa model...")
        face_res_model, upscaler_instance = load_models_cached(
            face_restoration_type, model_upscaler_key, upscaler_tile, upscaler_tile_overlap, upscaler_half, progress=progress
        )
        print("Model siap.")
    except Exception as e:
        raise gr.Error(f"Gagal memuat model: {e}")

    img_original = input_image.convert("RGB")
    
    # 2. Downscaling
    print("Memulai downscaling gambar...")
    img_for_processing = downscale_image_by_factor(img_original.copy(), downscale_factor_value, progress=progress)
    img_processed = img_for_processing.copy()
    print("Downscaling selesai.")

    # 3. Restorasi Wajah
    if face_restoration_type != "Disabled" and face_res_model:
        print(f"Menerapkan restorasi wajah: {face_restoration_type}")
        try:
            progress(0, desc=f"Menerapkan restorasi wajah ({face_restoration_type})...")
            img_processed = process_face_restoration(
                img_processed,
                face_res_model,
                face_restoration_visibility,
                face_restoration_weight,
            )
            progress(1, desc="Restorasi wajah selesai.")
            print("Restorasi wajah selesai.")
        except Exception as e:
            print(f"Error selama restorasi wajah: {e}")
            gr.Warning(f"Restorasi wajah gagal: {e}. Melanjutkan dengan upscaling.")

    # 4. Upscaling
    print(f"Upscaling gambar dengan {model_upscaler_key} ke skala {scale_of_the_image_x}x")
    try:
        progress(0, desc=f"Upscaling gambar ({model_upscaler_key})...")
        upscaled_image = upscaler_instance.upscale(
            img_processed,
            scale_of_the_image_x,
        )
        progress(1, desc="Upscaling selesai.")
        print("Upscaling selesai.")
    except Exception as e:
        raise gr.Error(f"Upscaling gagal: {e}")

    # 5. Enkripsi
    print("Mengenkrpsi gambar...")
    try:
        encrypted_data = encrypt_image(upscaled_image, encryption_password)
        
        os.makedirs(KAGGLE_OUTPUT_DIR, exist_ok=True)
        file_name = f"encrypted_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.enc"
        encrypted_file_path = os.path.join(KAGGLE_OUTPUT_DIR, file_name)
        with open(encrypted_file_path, "wb") as f:
            f.write(encrypted_data)
        print(f"File terenkripsi disimpan di: {encrypted_file_path}")

        return ([img_for_processing, upscaled_image], encrypted_data)
    except Exception as e:
        raise gr.Error(f"Enkripsi dan penyimpanan gagal: {e}")


# Fungsi untuk mendekripsi dan membuat tautan unduhan
def decrypt_and_create_download_link(encrypted_data_state: Optional[bytes], password: str):
    """Mendekripsi dari state dan membuat tautan unduhan Base64. MEMORY-ONLY DECRYPTION."""
    if encrypted_data_state is None:
        return gr.update(value="<p style='color: red;'>Tidak ada data terenkripsi untuk didekripsi. Proses dulu.</p>", visible=True), gr.update(interactive=False)
    
    try:
        print("Mendekripsi data dari memori untuk pengunduhan...")
        
        # 1. Dekripsi
        decrypted_image = decrypt_image(encrypted_data_state, password)
        
        # 2. Simpan gambar yang didekripsi ke buffer memori sebagai JPEG
        img_buffer = BytesIO()
        decrypted_image.save(img_buffer, "JPEG")
        img_bytes = img_buffer.getvalue()
        
        # 3. Encode ke Base64
        base64_data = base64.b64encode(img_bytes).decode('utf-8')
        
        # 4. Buat tautan unduhan HTML (KURUNG KURAWAL DILOLOSKAN)
        href_string = f"""
        <style>
        .download-link {{
            padding: 10px;
            background-color: #4CAF50; /* Green */
            color: white !important;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            border-radius: 8px;
            font-size: 16px;
            margin-top: 10px;
        }}
        .download-link:hover {{
            background-color: #45a049;
        }}
        </style>
        <a href="data:image/jpeg;base64,{base64_data}" download="upscaled_image.jpeg" class="download-link">✅ Klik di sini untuk Unduh Gambar Terdekripsi (JPEG)</a>
        """
        
        return gr.update(value=href_string, visible=True), gr.update(interactive=True)
    except ValueError as e:
        error_msg = str(e)
        gr.Warning(f"Gagal mendekripsi: {error_msg}")
        return gr.update(value=f"<p style='color: red;'>Gagal mendekripsi. {error_msg}.</p>", visible=True), gr.update(interactive=False)
    except Exception as e:
        gr.Warning(f"Error tak terduga selama dekripsi: {e}")
        return gr.update(value=f"<p style='color: red;'>Error tak terduga saat membuat tautan unduhan: {e}</p>", visible=True), gr.update(interactive=False)

# Fungsi untuk memperbarui tampilan dimensi
def update_dimensions_display(input_image: Optional[Image.Image], downscale_factor_value: float) -> str:
    """Menampilkan dimensi asli dan dimensi setelah downscale."""
    if input_image is None:
        return "Unggah gambar untuk melihat dimensi."
    
    try:
        original_width, original_height = input_image.size
        safe_factor = max(0.0, min(1.0, downscale_factor_value))
        calculated_width, calculated_height = get_calculated_dimensions(original_width, original_height, safe_factor)
        
        return (
            f"Dimensi Asli: {original_width}x{original_height}px\n"
            f"Faktor Skala Input: {safe_factor:.2f}\n"
            f"Dimensi Setelah Skala Input: {calculated_width}x{calculated_height}px"
        )
    except Exception as e:
        return f"Error membaca gambar: {e}"

# --- Bagian 4: Definisi Antarmuka Gradio ---
with gr.Blocks(css=".download-link {text-decoration: none;}") as demo:
    gr.Markdown("# 🛡️ Peningkat Gambar dan Restorasi Wajah yang Sangat Aman")
    gr.Markdown(f"Aplikasi ini memproses gambar Anda, mengenkripsinya dengan **AES-256 GCM** menggunakan kunci yang berasal dari kata sandi Anda melalui **PBKDF2**, dan **menyimpan versi terenkripsi di disk** di `{KAGGLE_OUTPUT_DIR}`. Versi terdekripsi hanya dibuat dan disajikan di memori saat Anda mengunduh.")
    gr.Markdown(f"Saat ini berjalan di **{cl_device.upper()}**.")

    # State untuk menyimpan data terenkripsi di antara tombol
    encrypted_data_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Pengaturan Input & Proses")
            
            password_input = gr.Textbox(
                label="Kata Sandi Enkripsi (Wajib)",
                type="password",
                interactive=True,
                placeholder="Masukkan kata sandi yang kuat di sini..."
            )
            image_input = gr.Image(type="pil", label="Gambar Input", elem_id="input_image")
            
            # Pengaturan Downscale (Input)
            downscale_factor_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                label="Faktor Skala Input (Downscale Awal)",
                info="Gunakan < 1.0 untuk simulasi low-res, 1.0 untuk resolusi asli.",
                value=1.0,
                interactive=True
            )
            dim_display_textbox = gr.Textbox(
                label="Informasi Dimensi Pemrosesan", 
                interactive=False, 
                lines=3
            )
            
            gr.Markdown("## ⚙️ Pengaturan Upscale & Restorasi")
            
            model_upscaler_dropdown = gr.Dropdown(
                choices=list(UPSCALER_DICT_GUI.keys()), 
                label="Model Upscaler", 
                value="RealESRNet_x4plus"
            )
            scale_slider = gr.Slider(
                minimum=1, 
                maximum=8, 
                step=0.1, 
                label="Faktor Skala Upscale Akhir (x)", 
                value=4
            )
            
            with gr.Accordion("Pengaturan Lanjutan (Tiling & Presisi)", open=False):
                upscaler_tile_slider = gr.Slider(minimum=0, maximum=512, step=16, label="Ukuran Tile Upscaler (0=nonaktif)", value=192)
                upscaler_tile_overlap_slider = gr.Slider(minimum=0, maximum=48, step=1, label="Tumpang Tindih Tile Upscaler", value=8)
                # PERUBAHAN: Default upscaler_half diubah menjadi False
                upscaler_half_checkbox = gr.Checkbox(
                    label="Upscaler Presisi Setengah (Mempercepat di GPU, dapat mengurangi memori)", 
                    value=False, # Diubah dari cl_device == "cuda"
                    interactive=cl_device == "cuda" # Interaktif hanya di CUDA
                )
                
            face_restoration_dropdown = gr.Dropdown(
                choices=["Disabled", "CodeFormer", "GFPGAN", "RestoreFormer"], 
                label="Restorasi Wajah", 
                value="GFPGAN"
            )
            face_restoration_visibility_slider = gr.Slider(minimum=0, maximum=1, step=0.01, label="Visibilitas Restorasi Wajah", value=0.6)
            face_restoration_weight_slider = gr.Slider(minimum=0, maximum=1, step=0.01, label="Bobot Restorasi Wajah", value=0.5)

            submit_button = gr.Button("1. 🚀 Proses Gambar & Enkripsi", variant="primary")
            download_button = gr.Button("2. ⬇️ Unduh Gambar Terdekripsi", interactive=False, variant="secondary")

        with gr.Column(scale=2):
            gr.Markdown("## 🖼️ Hasil")
            image_slider_output = gr.ImageSlider(
                label="Perbandingan: Input Awal (setelah downscale) vs. Hasil Upscale",
                show_label=True,
                value=None,
                format="jpeg" 
            )
            download_link_html = gr.HTML(
                value="<p>Setelah proses selesai, tombol 'Unduh' akan aktif.</p>", 
                visible=True
            )
            
    # --- Interaksi Dinamis ---
    inputs_for_dim_update = [image_input, downscale_factor_slider]

    for control in inputs_for_dim_update:
        control.change(
            fn=update_dimensions_display,
            inputs=inputs_for_dim_update,
            outputs=dim_display_textbox,
            queue=False
        )

    # Logika Tombol Proses
    submit_button.click(
        fn=process_and_encrypt,
        inputs=[
            image_input,
            downscale_factor_slider,
            model_upscaler_dropdown,
            scale_slider,
            upscaler_tile_slider,
            upscaler_tile_overlap_slider,
            face_restoration_dropdown,
            face_restoration_visibility_slider,
            face_restoration_weight_slider,
            upscaler_half_checkbox,
            password_input,
        ],
        outputs=[image_slider_output, encrypted_data_state]
    ).success(
        fn=lambda: [gr.update(interactive=True), gr.update(value="<p style='color: green;'>✅ Proses selesai. Data terenkripsi disimpan. Masukkan kata sandi lagi dan klik Unduh.</p>", visible=True)],
        inputs=[],
        outputs=[download_button, download_link_html]
    )
    
    # Logika Tombol Unduh
    download_button.click(
        fn=decrypt_and_create_download_link,
        inputs=[encrypted_data_state, password_input],
        outputs=[download_link_html, download_button],
        queue=False 
    )

# --- Bagian 5: Luncurkan Aplikasi Gradio ---
if __name__ == "__main__":
    try:
        from Crypto.Cipher import AES
        import requests
    except ImportError as e:
        print(f"Pustaka penting tidak ditemukan: {e}. Menginstal 'pycryptodomex' dan 'requests'...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pycryptodomex requests"])
            print("Instalasi berhasil. Silakan jalankan ulang skrip.")
            sys.exit(0)
        except Exception as install_e:
            print(f"Gagal menginstal dependensi: {install_e}")
            sys.exit(1)
    
    with demo:
        dim_display_textbox.value = update_dimensions_display(None, downscale_factor_slider.value)
        
    demo.launch(debug=True, inline=False)
