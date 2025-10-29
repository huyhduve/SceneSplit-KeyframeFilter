import torch 
from transnetv2_pytorch import TransNetV2

class SceneDetectorTransNet:
    def __init__(self, device='auto'):
        self.device = device
        self.model = TransNetV2().to(self.device)
        self.model.eval()
        state_dict = torch.load("transnetv2-pytorch-weights.pth", map_location=self.device)
        self.model.load_state_dict(state_dict)

    def detect_scenes(self, video_path):
        with torch.no_grad():
            scenes = self.model.detect_scenes(video_path)
        
        return scenes


if __name__ == "__main__":
    detector = SceneDetectorTransNet(device='cuda')
    video_file = r"D:\Python\EXTRACTING-FILTER-FRAMES\examples\video1.mp4"
    scenes = detector.detect_scenes(video_file)
    print("Detected scenes:", scenes)

