import cv2 
import os

class VideoExtract:
    def __init__(self, vid_path, time_stamp, output_dir):
        self.vidpath = vid_path
        self.time_stamp = time_stamp
        self.output_dir = output_dir


    def process(self, frame_dist):
        shot = []
        with open(self.time_stamp, "r") as shot_f:
            for line in shot_f:
                start, end = map(int, line.split())
                shot.append((start, end))
    
        vidname = os.path.basename(self.vidpath).split(".")[0]
        vidfolder_path = os.path.join(self.output_dir, vidname)

        if not os.path.exists(vidfolder_path):
            os.makedirs(vidfolder_path)
        
        cap = cv2.VideoCapture(self.vidpath)
        for start, end in shot:
            for frame_idx in range(start, end, frame_dist):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                frame_path = os.path.join(vidfolder_path, f"{frame_idx}.webp")
                if ret :
                    cv2.imwrite(frame_path, frame)
                    print("[INFO] Saved",frame_idx)
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    TIMESTAMP = r"D:\Python\EXTRACTING-FILTER-FRAMES\outputs\scenes\video1.txt"
    VIDEO_PATH = r"D:\Python\EXTRACTING-FILTER-FRAMES\examples\video1.mp4"
    OUTPUT_DIR = r"D:\Python\EXTRACTING-FILTER-FRAMES\outputs\frame"

    extractor = VideoExtract(VIDEO_PATH, TIMESTAMP, OUTPUT_DIR)
    extractor.process(5)
    
