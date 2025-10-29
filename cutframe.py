#có lọc nhiễu
import cv2
import os
import logging
import torch
import torch.nn.functional as F
import open_clip
import glob
import numpy as np
import faiss
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# ===== Config =====
VIDEO_DIR   = r"D:\Python\L22_a\K08"       # thư mục chứa video
SCENES_DIR  = r"D:\Python\L22_a\scenes\K08"      # thư mục chứa timestamp
OUTPUT_DIR  = r"D:\Python\data\Keyframes\K08"  # thư mục output
assert Path(VIDEO_DIR).exists(), f"[ERROR] VIDEO_DIR not found: {VIDEO_DIR}"
assert Path(SCENES_DIR).exists(), f"[ERROR] SCENES_DIR not found: {SCENES_DIR}"

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class KeyframeExtractor:
    def __init__(
        self,
        video_path: str,
        timestamp_path: str,
        video_files: list,
        timestamp_files: list,
        width: int,
        height: int,
        output_dir: str = 'keyframes',
        max_workers: int = 4,
        sample_rate: int = 5,
        similarity_threshold: float = 0.85,
        ssim_threshold: float = 0.9,
    ):
        # Ensure matching counts
        assert len(video_files) == len(timestamp_files), \
            "Number of video files must match number of timestamp files."

        # Paths and file lists
        self.video_path = video_path
        self.timestamp_path = timestamp_path
        self.video_files = video_files
        self.timestamp_files = timestamp_files
        self.width = width
        self.height = height
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.sample_rate = sample_rate
        self.similarity_threshold = similarity_threshold
        self.ssim_threshold = ssim_threshold

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print("Loading CLIP ViT-L/14 model via open_clip")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K', device=device
        )
        self.model.eval()
        self.model = self.model.half() #FP16
        self.use_fp16 = True
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode='reduce-overhead')

        print("Prepare output directory")
        self._prepare_output_dir()

    def _prepare_output_dir(self):
        # chỉ tạo thư mục gốc, không xoá dữ liệu cũ
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def extract_keyframes(self):

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for vid_file, ts_file in zip(self.video_files, self.timestamp_files):
                futures.append(executor.submit(self._process_video, vid_file, ts_file))

            for future in as_completed(futures):
                result = future.result()
                if result:
                    logging.info(result)

    def _process_video(self, video_path: str, timestamp_path: str) -> str:

        rel_path = Path(video_path).relative_to(self.video_path)
        out_dir = self.output_dir / rel_path.parent / rel_path.stem
        video_name = rel_path.stem

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Cannot open video file: {video_path}")
            return ""

        ok, frame = False, None
        for fid in range(1, 23456, 1000):  # n-kfps
          cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
          ok, frame = cap.read()
          if ok and frame is not None:
              break

        if not ok:
            logging.error(f"Cannot read any valid frame from: {video_path}")
            return ""

        # đọc danh sách shots
        shots = []
        with open(timestamp_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        start, end = map(int, parts)
                        shots.append((start, end))
                    except ValueError:
                        continue

        if not shots:
            logging.warning(f"No valid shots in {timestamp_path}")
            cap.release()
            return f"Skipped {video_name} (no shots)"

        resume_shot_idx = 0
        resume_start = None
        existing_frames = set()

        if out_dir.exists():
            existing_frames = {int(p.stem) for p in out_dir.glob("*.webp")}
            if existing_frames:
                last_frame = max(existing_frames)

                # Tìm shot chứa last_frame
                found = False
                for i, (s, e) in enumerate(shots):
                    if s <= last_frame <= e:
                        resume_shot_idx = i
                        resume_start = last_frame + 1
                        found = True
                        break
                if not found and last_frame > shots[-1][1]:
                    # đã xử lý hết video
                    logging.info(f"Skip {video_name}, already fully processed.")
                    cap.release()
                    return f"Skipped {video_name}"
        else:
            out_dir.mkdir(parents=True, exist_ok=True)


        for idx, (start, end) in enumerate(shots):
            if idx < resume_shot_idx:
                continue

            if idx == resume_shot_idx and resume_start is not None:
                if resume_start > end:
                    continue  # shot này đã xong
                start = resume_start  # bắt đầu lại từ frame cuối + 1

            shot_len = end - start + 1
            adaptive_rate = self.sample_rate
            if shot_len > 969:
                adaptive_rate *= 3
            elif shot_len > 333:
                adaptive_rate *= 2
            elif shot_len > 36:
                adaptive_rate += 1

            frame_idxs = list(range(start, end + 1, adaptive_rate))

            tensor_list = []
            valid_frame_idxs = []
            for fid in frame_idxs:
                #print(f"frame {fid} in {video_name}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"Cannot read frame {fid} in {video_name}")
                    continue

                # lọc junk
                if is_junk_frame(frame):
                    continue

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = self.preprocess(Image.fromarray(img)).unsqueeze(0)
                tensor_list.append(tensor)
                valid_frame_idxs.append(fid)

            if not tensor_list:
                logging.warning(f"No frames sampled for shot {start}-{end} in {video_name}")
                continue

            batch_size = 1024
            all_embs = []
            for i in range(0, len(tensor_list), batch_size):
                batch = torch.cat(tensor_list[i:i+batch_size]).to(self.device)
                if self.use_fp16:
                    batch = batch.half()
                with torch.no_grad():
                    emb = self.model.encode_image(batch)
                all_embs.append(emb.cpu())
                del batch, emb
                #torch.cuda.empty_cache()

            embs = torch.cat(all_embs).to(self.device)

            # chọn số cụm
            n_samples = len(embs)
            if n_samples < 150:
                K = 1
            else:
                K = max(1, min(n_samples // 236, 12)) # 1/n, max m

            embs_np = embs.detach().float().cpu().numpy() # float32 cho faiss
            faiss.normalize_L2(embs_np)

            d = embs_np.shape[1]
            use_gpu = faiss.get_num_gpus() > 0  #gpu->true, window_cpu

            #cluster
            kmeans = faiss.Kmeans(d, K, niter=20, verbose=False, gpu=use_gpu)
            kmeans.train(embs_np)

            #centroid
            D, I = kmeans.index.search(embs_np, 1) # D: khoảng cách, I: index cụm
            labels = I.ravel()

            chosen_frames = []
            for k in range(K):
                idxs = np.where(labels == k)[0]
                if idxs.size == 0:
                    continue
                best_idx = idxs[np.argmin(D[idxs, 0])]
                chosen_frames.append(valid_frame_idxs[best_idx])

            frame_cache = {}
            for fnum in chosen_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
                ret, frm = cap.read()
                if not ret: continue
                gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
                frame_cache[fnum] = gray

            # lọc trùng lặp
            idx_map = {fid: i for i, fid in enumerate(valid_frame_idxs)}
            final_frames = []
            for fnum in chosen_frames:
                keep = True
                emb_f = embs[idx_map[fnum]]
                gray_f = frame_cache[fnum]
                for g in final_frames:
                    gray_g = frame_cache[g]
                    s = ssim(gray_f, gray_g, data_range=255)
                    if s > self.ssim_threshold:
                            keep = False
                            break

                    emb_g = embs[idx_map[g]]
                    sim = F.cosine_similarity(emb_f.unsqueeze(0), emb_g.unsqueeze(0)).item()
                    if sim > self.similarity_threshold:
                            keep = False
                            break
                if keep:
                    final_frames.append(fnum)

            # lưu keyframes
            for fnum in final_frames:
                print(f"frame {fnum} in {video_name}")
                if fnum in existing_frames:
                    logging.info(f"Skip frame {fnum} (already exists) in {video_name}")
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
                ret, best_frame = cap.read()
                if ret:
                    resized = cv2.resize(best_frame, (self.width, self.height))
                    out_path = out_dir / f"{fnum}.webp"
                    cv2.imwrite(str(out_path), resized)
                else:
                    logging.warning(f"Failed to read frame {fnum} from {video_name}")

            del all_embs, embs
            torch.cuda.empty_cache()
            logging.info(f"Freed GPU memory after shot {start}-{end} in {video_name}")

        cap.release()
        #torch.cuda.empty_cache()
        return f"Finished processing {video_name}"

def is_junk_frame(frame, mean_thresh=(10, 245), entropy_thresh=1.5, blur_thresh=50, color_ratio_thresh=0.95, bins=32):
    """
    Trả về True nếu frame là junk
    """
    # chuyển về xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Lọc frame tối hoặc sáng
    mean_val = gray.mean()
    if mean_val < mean_thresh[0] or mean_val > mean_thresh[1]:
        return True

    # 1b. Lọc frame quá đồng nhất (sử dụng histogram để nhanh hơn)
    hist_b = cv2.calcHist([frame], [0], None, [bins], [0,256])
    hist_g = cv2.calcHist([frame], [1], None, [bins], [0,256])
    hist_r = cv2.calcHist([frame], [2], None, [bins], [0,256])

    max_ratio = max(hist_b.max(), hist_g.max(), hist_r.max()) / (frame.shape[0]*frame.shape[1])
    if max_ratio > color_ratio_thresh:
        return True

    # 2. Lọc frame ít thông tin (entropy thấp)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    hist /= hist.sum() + 1e-7
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    if entropy < entropy_thresh:
        return True

    # 3. Lọc frame mờ (blur)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    if fm < blur_thresh:
        return True

    return False

def get_video_and_timestamps(video_list, video_dir, timestamp_dir):
    vid_files = []
    ts_files = []
    for vid in video_list:
        rel = Path(vid).relative_to(video_dir).with_suffix(".scenes.txt")
        ts_file = Path(timestamp_dir) / rel

        #print("Mapping:", vid, "->", ts_file)
        if not ts_file.exists():
            logging.warning(f"Skip video (no scenes): {vid}")
            continue
        if not Path(vid).exists():
            logging.warning(f"Skip timestamp (no video): {ts_file}")
            continue

        vid_files.append(vid)
        ts_files.append(str(ts_file))
    return vid_files, ts_files

def get_videos_to_process(video_dir, output_dir):

    all_videos = sorted(glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True))
    return all_videos


if __name__ == "__main__":

    VID_PATH        = VIDEO_DIR
    TIMESTAMP_PATH  = SCENES_DIR

    all_videos = get_videos_to_process(VID_PATH, OUTPUT_DIR)
    print(f"Found {len(all_videos)} videos to check/process.")

    VID_FILES, TIMESTAMP_FILES = get_video_and_timestamps(all_videos, VID_PATH, TIMESTAMP_PATH)

    print("[+] Operation probably taking long time, be patient")
    print("[+] Processing....")

    extractor = KeyframeExtractor(
        VID_PATH,
        TIMESTAMP_PATH,
        VID_FILES,
        TIMESTAMP_FILES,
        width=640,
        height=360,
        output_dir=OUTPUT_DIR,
        max_workers=6,
        sample_rate=5,
    )
    extractor.extract_keyframes()