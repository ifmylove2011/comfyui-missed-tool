from .trim_bg import TrimBG,TrimBGAdvanced
from .img_queue_load import ImageQueueLoader
from .img_load import LoadImageA

NODE_CLASS_MAPPINGS = {
    "TrimBG": TrimBG,
    "TrimBGAdvanced": TrimBGAdvanced,
    "ImageQueueLoader": ImageQueueLoader,
    "LoadImageA": LoadImageA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrimBG": "TrimBG",
    "TrimBGAdvanced": "TrimBG Advanced",
    "ImageQueueLoader": "Image Queue Loader",
    "LoadImageA": "Load Image Alpha",
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']