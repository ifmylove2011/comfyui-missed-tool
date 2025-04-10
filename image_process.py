import math


class ScaleMultilplePixels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "position": (
                    ["top-left", "top-center", "top-right", "right-center", "bottom-right", "bottom-center",
                     "bottom-left",
                     "left-center", "center"], {"default": "center"}),
                "base_pixels": ("INT", {"default": 8, "min": 1, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "missed-tool"

    def execute(self, image, base_pixels, position):
        _, oh, ow, _ = image.shape

        width = math.floor(ow / base_pixels) * base_pixels
        height = math.floor(oh / base_pixels) * base_pixels

        x = 0
        y = 0

        if "center" in position:
            x = round((ow - width) / 2)
            y = round((oh - height) / 2)
        if "top" in position:
            y = 0
        if "bottom" in position:
            y = oh - height
        if "left" in position:
            x = 0
        if "right" in position:
            x = ow - width

        x2 = x + width
        y2 = y + height

        if x2 > ow:
            x2 = ow
        if x < 0:
            x = 0
        if y2 > oh:
            y2 = oh
        if y < 0:
            y = 0

        image = image[:, y:y2, x:x2, :]

        return (image,)
