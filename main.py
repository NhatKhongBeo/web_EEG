from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from signal_processing.split_data import split_segments_to_queue

import os
import mne
import numpy as np

# Tạo ứng dụng FastAPI
app = FastAPI()

# Cấu hình thư mục lưu file
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Tích hợp các file tĩnh (CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Cấu hình thư mục template
templates = Jinja2Templates(directory="templates")

SUPPORTED_FORMATS = (".fif", ".edf", ".bdf", ".gdf", ".vhdr", ".eeg", ".set")

# Biến toàn cục để lưu đường dẫn file (chỉ dành cho demo)
file_path_global = ""


@app.get("/", response_class=HTMLResponse)
async def render_upload_page(request: Request):
    """
    Endpoint hiển thị giao diện tải lên file.
    """
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload-eeg/")
async def upload_eeg(file: UploadFile = File(...)):
    """
    Endpoint xử lý tải lên, chọn 4 kênh EEG, normalize tín hiệu và lưu file.
    """
    global file_path_global

    # Kiểm tra định dạng file
    if not file.filename.endswith(SUPPORTED_FORMATS):
        return JSONResponse(
            content={
                "error": f"Unsupported file format. Allowed formats: {', '.join(SUPPORTED_FORMATS)}"
            },
            status_code=400,
        )

    # Lưu file vào thư mục tạm
    temp_path = os.path.join(UPLOAD_DIR, f"temp_{file.filename}")
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        # Đọc file EEG bằng mne
        raw = mne.io.read_raw(temp_path, preload=True)
        info = raw.info
        duration = raw.times[-1]
        channels = len(info["ch_names"])
        sampling_rate = info["sfreq"]

        # Lấy danh sách các kênh EEG
        eeg_channels = [ch for ch in raw.info['ch_names'] if 'EEG' in ch]
        if len(eeg_channels) < 4:
            return JSONResponse(
                content={"error": "The file does not have enough EEG channels (minimum 4 required)."},
                status_code=400,
            )

        # Chọn 4 kênh đầu tiên
        selected_channels = eeg_channels[:4]
        raw = raw.pick_channels(selected_channels)

        # Áp dụng normalization
        raw_normal = raw.copy().apply_function(lambda x: (x - np.mean(x)) / np.std(x))

        # Lưu lại file normalized dưới dạng .fif
        normalized_file_path = os.path.join(UPLOAD_DIR, f"normalized_{os.path.splitext(file.filename)[0]}.fif")
        raw_normal.save(normalized_file_path, overwrite=True)

        # Lưu đường dẫn normalized file
        file_path_global = normalized_file_path

        # Tạo ghi chú cho người dùng
        notes = {
            "filename": file.filename,
            "normalized_filepath": normalized_file_path,
            "selected_channels": selected_channels,
            "duration_seconds": f"{duration:.2f}",
            "channels_count": len(selected_channels),
            "sampling_rate_hz": f"{sampling_rate:.2f}",
        }

        return {"notes": notes}
    except Exception as e:
        return JSONResponse(
            content={"error": f"Error processing file: {str(e)}"},
            status_code=500,
        )
    finally:
        # Xóa file tạm sau khi xử lý
        if os.path.exists(temp_path):
            os.remove(temp_path)





@app.post("/detect/")
async def detect_signal():
    """
    Endpoint xử lý phát hiện tín hiệu EEG đã được normalize.
    """
    global file_path_global

    if not file_path_global:
        return JSONResponse(
            content={"error": "No normalized file found. Please upload and normalize a file first."},
            status_code=400,
        )

    try:
        # Đọc lại file normalized
        raw = mne.io.read_raw_fif(file_path_global, preload=True)

        # Lấy dữ liệu và tần số lấy mẫu
        data = raw.get_data()  # Dạng numpy array (channels x samples)
        sfreq = raw.info['sfreq']

        # Chia dữ liệu thành các phân đoạn và xếp vào queue
        queue = split_segments_to_queue(data, sfreq)

        # Trả về thông tin về số lượng phân đoạn
        return {
            "message": "File processed and segments added to queue",
            "num_segments": len(queue)
        }
    except Exception as e:
        return JSONResponse(
            content={"error": f"Error during detection: {str(e)}"},
            status_code=500,
        )

