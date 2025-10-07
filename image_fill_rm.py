import torch

NODE_NAME = 'ColorImageFillRm'


class ColorImageFillRm:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "color_src": (
                "STRING", {"default": "#ffffff,#ff0000", "multiline": False, "tooltip": "要替换的颜色，可以输入多个颜色，用逗号分隔"}),
                "color_replace": (
                "STRING", {"default": "#00ffff,#0000ff", "multiline": False, "tooltip": "对应的替换颜色，数量需与 color_src 相同"}),
                "rm_color": ("STRING", {"default": "#000000", "multiline": False, "tooltip": "要透明化的颜色"}),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "颜色匹配容差（欧氏距离）。0.05≈RGB每通道±13色阶的偏差。",
                    },
                ),
            }
        }


    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "color_image_fill_rm"
    CATEGORY = "missed-tool"


    def _hex_to_rgb(self, hex_str):
        hex_str = hex_str.strip().lstrip("#")
        return tuple(int(hex_str[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


    def color_image_fill_rm(self, images, color_src, color_replace, rm_color, threshold):
        imgs = images.clone()
        b, h, w, c = imgs.shape

        # 若没有 alpha 通道则添加
        if c == 3:
            alpha = torch.ones((b, h, w, 1), device=imgs.device)
            imgs = torch.cat([imgs, alpha], dim=-1)

        rm_rgb = torch.tensor(self._hex_to_rgb(rm_color), device=imgs.device)

        color_src_list = [self._hex_to_rgb(c) for c in color_src.split(",")]
        color_replace_list = [self._hex_to_rgb(c) for c in color_replace.split(",")]

        if len(color_src_list) != len(color_replace_list):
            raise ValueError("color_src 和 color_replace 的数量必须相同。")

        # 多颜色替换
        for src_rgb, rep_rgb in zip(color_src_list, color_replace_list):
            src_rgb = torch.tensor(src_rgb, device=imgs.device)
            rep_rgb = torch.tensor(rep_rgb, device=imgs.device)

            diff = torch.sqrt(torch.sum((imgs[..., :3] - src_rgb) ** 2, dim=-1, keepdim=True))
            mask = (diff < threshold).float()

            imgs[..., :3] = imgs[..., :3] * (1 - mask) + rep_rgb * mask

        # 透明化处理
        diff_rm = torch.sqrt(torch.sum((imgs[..., :3] - rm_rgb) ** 2, dim=-1, keepdim=True))
        mask_rm = (diff_rm < threshold).float()
        imgs[..., 3] *= (1 - mask_rm).squeeze(-1)

        return (imgs,)
