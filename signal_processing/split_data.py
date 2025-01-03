from collections import deque
import numpy as np
def split_segments_to_queue(data: np.ndarray, sfreq: float, segment_duration: int = 30):
    """
    Chia tín hiệu thành các phân đoạn và xếp vào queue.

    Parameters:
        data (numpy.ndarray): Tín hiệu EEG dạng (channels x samples).
        sfreq (float): Tần số lấy mẫu của tín hiệu.
        segment_duration (int): Thời lượng của mỗi phân đoạn (giây).

    Returns:
        deque: Hàng đợi chứa các phân đoạn tín hiệu (numpy arrays).
    """
    segment_samples = int(segment_duration * sfreq)  # Số mẫu trong một phân đoạn
    num_segments = data.shape[1] // segment_samples  # Số phân đoạn

    queue = deque()  # Tạo hàng đợi để lưu các phân đoạn

    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = data[:, start:end]  # Lấy phân đoạn từ dữ liệu gốc
        queue.append(segment)  # Thêm phân đoạn vào queue

    return queue
