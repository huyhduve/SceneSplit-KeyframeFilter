import torch 
torch.use_deterministic_algorithms(False)
import os, warnings
warnings.filterwarnings("ignore")
from transnetv2_pytorch import TransNetV2


class SceneDetectorTransNet:
    def __init__(self, device='auto'):
        self.device = device
        self.model = TransNetV2().to(self.device)
        self.model.eval()
        weights_path = os.path.join(os.path.dirname(__file__),
                                    'transnetv2-pytorch-weights.pth')
        
        state_dict = torch.load(weights_path)
        self.model.load_state_dict(state_dict)

    def detect_scenes(self, video_path):
        with torch.no_grad():
            scenes = self.model.detect_scenes(video_path)
        
        return scenes


if __name__ == "__main__":
    detector = SceneDetectorTransNet(device='cuda')
    video_file = r"D:\Python\EXTRACTING-FILTER-FRAMES\examples\video1.mp4"
    output_file = r"D:\Python\EXTRACTING-FILTER-FRAMES\outputs\video1.txt"
    scenes = detector.detect_scenes(video_file)
    
    with open(output_file, "w") as f:
        for scene in scenes:
            f.write( f'{scene["start_frame"]} {scene["end_frame"]}\n')
    

