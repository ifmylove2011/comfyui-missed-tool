from .trim_bg import TrimBG,TrimBGAdvanced
from .img_queue_load import ImageQueueLoader
from .img_load import LoadImageA
from .text_file_save import TxtSave
from .image_process import ScaleMultilplePixels

NODE_CLASS_MAPPINGS = {
    "TrimBG": TrimBG,
    "TrimBGAdvanced": TrimBGAdvanced,
    "ImageQueueLoader": ImageQueueLoader,
    "LoadImageA": LoadImageA,
    "TxtSave": TxtSave,
    "ScaleMultilplePixels": ScaleMultilplePixels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrimBG": "TrimBG",
    "TrimBGAdvanced": "TrimBG Advanced",
    "ImageQueueLoader": "Image Queue Loader",
    "LoadImageA": "Load Image Alpha",
    "TxtSave": "Save Txt File",
    "ScaleMultilplePixels": "Scale Image to Multilple Pixels",
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']