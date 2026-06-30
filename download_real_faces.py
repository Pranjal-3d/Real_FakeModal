import os
import cv2
from huggingface_hub import hf_hub_download, list_repo_files

def download_and_extract_hf_videos():
    repo_id = "UniqueData/web-camera-face-liveness-detection"
    repo_type = "dataset"
    
    real_dir = "real"
    screen_dir = "screen"
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(screen_dir, exist_ok=True)

    try:
        files = list_repo_files(repo_id, repo_type=repo_type)
        real_indices = sorted({
            int(f.split("/")[-1].replace(".mp4", ""))
            for f in files if f.startswith("files/real/") and f.endswith(".mp4")
        })
        monitor_indices = sorted({
            int(f.split("/")[-1].replace(".mp4", ""))
            for f in files if f.startswith("files/monitor/") and f.endswith(".mp4")
        })
    except Exception:
        real_indices = list(range(20))
        monitor_indices = list(range(20))
    
    print(f"Found {len(real_indices)} real videos and {len(monitor_indices)} monitor videos on HuggingFace.")
    print("Starting download of real-world webcam face and screen videos...")
    
    # 1. Download and process REAL (live) face videos
    for idx in real_indices:
        filename = f"files/real/{idx}.mp4"
        try:
            print(f"Downloading {filename}...")
            # Download file from HF dataset hub
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type
            )
            
            # Extract frames
            cap = cv2.VideoCapture(local_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                print(f"Warning: frame count is 0 for {filename}, skipping frame extraction.")
                continue
                
            frame_indices = [int(i * total_frames / 15) for i in range(15)]
            saved_count = 0
            
            for f_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if ret:
                    out_path = os.path.join(real_dir, f"hf_real_{idx:03d}_{f_idx:03d}.jpg")
                    cv2.imwrite(out_path, frame)
                    saved_count += 1
            cap.release()
            print(f"Extracted {saved_count} frames from {filename} to real/")
        except Exception as e:
            print(f"Error handling {filename}: {e}")
            
    # 2. Download and process MONITOR (screen recapture) face videos
    for idx in monitor_indices:
        filename = f"files/monitor/{idx}.mp4"
        try:
            print(f"Downloading {filename}...")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type
            )
            
            cap = cv2.VideoCapture(local_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                print(f"Warning: frame count is 0 for {filename}, skipping frame extraction.")
                continue
                
            frame_indices = [int(i * total_frames / 15) for i in range(15)]
            saved_count = 0
            
            for f_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if ret:
                    out_path = os.path.join(screen_dir, f"hf_monitor_{idx:03d}_{f_idx:03d}.jpg")
                    cv2.imwrite(out_path, frame)
                    saved_count += 1
            cap.release()
            print(f"Extracted {saved_count} frames from {filename} to screen/")
        except Exception as e:
            print(f"Error handling {filename}: {e}")
            
    print("\nSuccessfully finished downloading and extracting face liveness test set!")

if __name__ == "__main__":
    download_and_extract_hf_videos()
