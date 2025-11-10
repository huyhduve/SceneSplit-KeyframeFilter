import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import os 
import shutil
from tqdm import tqdm
from glob import glob

class Filter: 
    def __init__(self):
        pass

    # reference: https://www.geeksforgeeks.org/computer-vision/algorithms-for-image-comparison/#what-is-image-comparison
    def histogram_compare(self, imageA, imageB, method='correlation'):
        imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
        imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)

        # Calculate histograms
        histA = cv2.calcHist([imageA], [0, 1], None, [50, 60], [0, 180, 0, 256])
        histB = cv2.calcHist([imageB], [0, 1], None, [50, 60], [0, 180, 0, 256])

        # Normalize histograms
        cv2.normalize(histA, histA, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(histB, histB, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Use correlation or other methods
        methods = {
            'correlation': cv2.HISTCMP_CORREL,
            'chi-square': cv2.HISTCMP_CHISQR,
            'bhattacharyya': cv2.HISTCMP_BHATTACHARYYA
        }

        comparison = cv2.compareHist(histA, histB, methods[method])
        return comparison
    
    def ssim_compare(self, imageA, imageB):
        gray_a = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        s = ssim(gray_a, gray_b, data_range=225)
        return s
    
    def process(self, image_input, output_dir = None, 
            SSIM_THRESHOLD=0.885, HISTOGRAM_THRESHOLD=0.8155):
        
        extension = ["jpg", "png", "webp"]
        image_paths = []

        if os.path.isfile(image_input):
            image_paths.append(image_input)
        elif os.path.isdir(image_input):
            for ext in extension:
                image_paths.extend(glob(os.join(image_input, "*.", ext)))
        else:
            raise ValueError(f"Invalid image path {image_input}")
        
        image_paths.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        print(f"[INFO] Found {len(image_paths)} images")

        prev_path = None
        results_path = [image_paths[0]]

        for cur_path in tqdm(image_paths, desc="Comparing Frames"):
            if prev_path is None:
                prev_path = cur_path
                continue 
            imageA = cv2.imread(prev_path)
            imageB = cv2.imread(cur_path)

            hist = self.histogram_compare(imageA, imageB)
            s = self.ssim_compare(imageA, imageB)
            if s < SSIM_THRESHOLD and hist < HISTOGRAM_THRESHOLD:
                results_path.append(cur_path)
            prev_path = cur_path

        print(f"[INFO] Selected {len(results_path)} frames out of {len(image_paths)}.")

        if output_dir is not None: 
            if(not os.path.exists(output_dir)):
                os.makedirs(output_dir)
            for path in tqdm(results_path, desc=f"Copying Selected Frames to {output_dir.split(os.sep)[-1]}"):
                shutil.copy(path, output_dir)
        
        return results_path 

if __name__ == "__main__":



            
        

