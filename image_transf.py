import numpy as np
import torch
from PIL import Image
from torchvision import transforms

loader = transforms.Compose([
    transforms.ToTensor()])
unloader = transforms.ToPILImage()


def pil2tensor(image):
    return loader(image).unsqueeze(0)


def tensor2pil(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    return unloader(image)


def tensor_to_pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil_to_tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
