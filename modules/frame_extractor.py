import os
import cv2
import logging
import glob
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class FrameExtractor:
    def __init__(self, video_path, timestamp_path, video_files, timestamp_files, 
                 width=640, height=360, output_dir="frames", max_workers=4, sample_rate=5):
        self.video_path = video_path
        self.timestamp_path = timestamp_path
        self.video_files = video_files
        self.timestamp_files = timestamp_files
        self.width = width
        self.height = height
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.sample_rate = sample_rate

    def extract_frames_from_video(self, video_file, timestamp_file):
        video_path = os.path.join(self.video_path, video_file)
        timestamp_path = os.path.join(self.timestamp_path, timestamp_file)
        
        # Create output directory for this video
        video_name = os.path.splitext(video_file)[0]
        output_path = os.path.join(self.output_dir, video_name)
        os.makedirs(output_path, exist_ok=True)
        
        # Read timestamps from space-separated text file
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
            return video_name, 0

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames_info = []
        
        for shot_index, (start_frame, end_frame) in enumerate(shots):
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_positions = range(start_frame, end_frame, self.sample_rate)
            for frame_pos in frame_positions:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame
                frame = cv2.resize(frame, (self.width, self.height))
                
                # Save frame as WebP
                frame_filename = f"{frame_pos}.webp"
                frame_path = os.path.join(output_path, frame_filename)
                
                # Set WebP compression parameters (quality 80 provides good balance)
                params = [cv2.IMWRITE_WEBP_QUALITY, 80]
                cv2.imwrite(frame_path, frame, params)
                
                frames_info.append({
                    'frame_path': frame_path,
                    'frame_pos': frame_pos,
                    'shot_index': shot_index
                })
        
        cap.release()
        
        # Save frames info
        info_path = os.path.join(output_path, 'frames_info.json')
        with open(info_path, 'w') as f:
            json.dump(frames_info, f)
        
        return video_name, len(frames_info)

    def process_all_videos(self):
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for video_file, timestamp_file in zip(self.video_files, self.timestamp_files):
                future = executor.submit(self.extract_frames_from_video, video_file, timestamp_file)
                futures.append(future)
            
            results = []
            for future in futures:
                try:
                    video_name, frame_count = future.result()
                    results.append(f"Processed {video_name}: {frame_count} frames extracted")
                except Exception as e:
                    results.append(f"Error processing video: {str(e)}")
            
            return results

def get_videos_to_process(video_dir, output_dir):
    """Get all video files recursively from video directory"""
    all_videos = sorted(glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True))
    return all_videos

def get_video_and_timestamps(video_list, video_dir, timestamp_dir):
    """Map video files to their corresponding timestamp files"""
    vid_files = []
    ts_files = []
    for vid in video_list:
        rel = Path(vid).relative_to(video_dir).with_suffix(".scenes.txt")
        ts_file = Path(timestamp_dir) / rel

        if not ts_file.exists():
            logging.warning(f"Skip video (no scenes): {vid}")
            continue
        if not Path(vid).exists():
            logging.warning(f"Skip timestamp (no video): {ts_file}")
            continue

        vid_files.append(vid)
        ts_files.append(str(ts_file))
    return vid_files, ts_files

if __name__ == "__main__":
    VID_PATH = r"D:\Python\L22_a\K08"
    TIMESTAMP_PATH = r"D:\Python\L22_a\scenes\K08"
    OUTPUT_DIR = "D:\Python\L22_a\K08"
    
    # Get all videos recursively
    all_videos = get_videos_to_process(VID_PATH, OUTPUT_DIR)
    print(f"Found {len(all_videos)} videos to check/process.")
    
    # Map videos to their timestamp files
    video_files, timestamp_files = get_video_and_timestamps(all_videos, VID_PATH, TIMESTAMP_PATH)
    
    extractor = FrameExtractor(
        VID_PATH,
        TIMESTAMP_PATH,
        video_files,
        timestamp_files,
        width=640,
        height=360,
        output_dir=OUTPUT_DIR,
        max_workers=4,
        sample_rate=5
    )
    
    results = extractor.process_all_videos()
    for result in results:
        print(result)