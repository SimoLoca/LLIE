import torch
import torchvision.transforms.functional as F
from torchvision.io import read_image
from .DCE_Net import DCE_Net
import cv2


class DCE(object):
    def __init__(self, ckpt_path: str, device: str):
        super().__init__()
        self.dev = device
        self.model = DCE_Net()
        # Load model ckpt
        self.model.load_state_dict(torch.load(ckpt_path)['model'])
        self.model.to(self.dev)
        print("Model loaded...")

    @torch.inference_mode()
    def enhance(self, image_path: str, size: tuple = (256, 256), save: bool = True):
        # Load image
        image = read_image(image_path) / 255.0
        image = F.resize(image, size)
        x = image.unsqueeze(0).to(self.dev)

        enhanced, _ = self.model(x)
        enhanced = enhanced.squeeze().permute(1, 2, 0).cpu().numpy()
        if save:
            enhanced = cv2.resize(enhanced, size[::-1])
            enhanced = (enhanced * 255).clip(0, 255).astype("uint8")
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        return enhanced
        

    

    
