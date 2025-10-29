from glob import glob 
import os 
from tqdm import tqdm

SOURCE = r"D:\Python\data\Keyframes\K08\NOT UPDATED\frames"
all_imgs = glob(SOURCE + "/*.webp")

for img_path in tqdm(all_imgs):
    name = os.path.basename(img_path).split('.')[0].split('_')[1]
    os.replace(img_path, SOURCE + "\\" + name + ".webp")