from itertools import zip_longest

from PIL import ImageColor

from .image_transf import *

NODE_NAME = 'TrimBG'


class TrimBGAdvanced:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK", {"forceInput": False}),
                "padding": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1, }),
                "alpha_thresold": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1, }),
                "fill_rbg": ("STRING", {"default": "", "multiline": False, "tooltip": "rbg value, such as #ffffff"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "trim_bg"
    CATEGORY = "missing-tool"

    def trim_bg(self, images, mask, padding=0, alpha_thresold=0, fill_rbg=''):
        crop_images = []
        if min(images.shape) > 1:  # batch
            for image, m in zip_longest(images, mask, fillvalue=None):
                self.trim_bg_process(crop_images, image, m, padding, alpha_thresold, fill_rbg)
        else:  # list
            self.trim_bg_process(crop_images, images, mask, padding, alpha_thresold, fill_rbg)
        return crop_images

    def trim_bg_process(self, crop_images, image, m, padding, alpha_thresold, fill_rbg):
        tensor_img = image.squeeze()
        img_src = None
        img_alpha = None
        if min(tensor_img.shape) > 3:  # RGBA
            src = tensor_to_pil(tensor_img)
            img_src = src.convert('RGBA')
            img_alpha = img_src
        else:  # RGB
            if m is not None:
                image_rgb = tensor_to_pil(tensor_img).convert('RGBA')
                mask_p = tensor_to_pil(1. - m)
                # print(f"image is {image_rgb.size}, mask is {m.shape}, mask_p is {mask_p.size}")
                img_src = Image.composite(image_rgb, Image.new("RGBA", image_rgb.size), mask_p)
                img_alpha = img_src
            else:
                img_src = tensor_to_pil(tensor_img)
                image_l = img_src.convert('L')
                image_rgb = img_src.convert('RGBA')
                img_alpha = Image.composite(image_rgb, Image.new("RGBA", image_rgb.size), image_l)
        bbox = self.get_bbox(img_alpha, alpha_thresold, padding)
        img_dst = img_src.crop(bbox)
        if len(fill_rbg) > 6:
            image_np = np.array(img_dst)
            rgb = ImageColor.getrgb(fill_rbg)
            image_np[image_np[:, :, 3] == 0] = [rgb[0], rgb[1], rgb[2], 255]
            img_dst = Image.fromarray(image_np)
        crop_images.append(pil_to_tensor(img_dst))
        return crop_images

    def get_bbox(self, img, alpha_thresold, padding):
        # 获取透明度通道
        alpha = img.split()[-1]
        w, h = img.size
        if alpha_thresold != 0:
            alpha_array = np.array(alpha)
            # 计算边界框
            for i in range(w):
                for j in range(h):
                    if alpha_array[j, i] < alpha_thresold:
                        alpha_array[j, i] = 0
            alpha = Image.fromarray(np.uint8(alpha_array))
        x, y, x1, y1 = alpha.getbbox()
        x = max(0, x - padding)
        y = max(0, y - padding)
        x1 = min(w, x1 + padding)
        y1 = min(h, y1 + padding)
        bbox = x, y, x1, y1
        return bbox


class TrimBG(TrimBGAdvanced):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "trim_bg"
    CATEGORY = "img-process/TrimBG"

    def trim_bg(self, images):
        return super().trim_bg(images, None, 0, 0)
