from .trim_bg import TrimBG, TrimBGAdvanced
from .img_queue_load import ImageQueueLoader
from .img_load import LoadImageA
from .text_file_save import TxtSave, TextSplitToList
from .image_process import ScaleMultilplePixels
from .lora_merge import LoraLoader, LoraSaver, LoraMerger
from .image_fill_rm import ColorImageFillRm
from .face_similarity_check import FaceRecSimilarityTopK, InsightFaceSimilarityTopK, HybridFaceSimilarityTopK, FaceIdentitySimilarity
from .xy_plot import NODE_CLASS_MAPPINGS as XY_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as XY_NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {
    "TrimBG": TrimBG,
    "TrimBGAdvanced": TrimBGAdvanced,
    "ColorImageFillRm": ColorImageFillRm,
    "ImageQueueLoader": ImageQueueLoader,
    "LoadImageA": LoadImageA,
    "TxtSave": TxtSave,
    "TextSplitToList": TextSplitToList,
    "ScaleMultilplePixels": ScaleMultilplePixels,
    "FaceSimilarityTopK": FaceRecSimilarityTopK,
    "InsightFaceSimilarityTopK": InsightFaceSimilarityTopK,
    "HybridFaceSimilarityTopK": HybridFaceSimilarityTopK,
    "FaceIdentitySimilarity": FaceIdentitySimilarity,
    "LoraLoad": LoraLoader,
    "LoraMerge": LoraMerger,
    "LoraSaver": LoraSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrimBG": "TrimBG",
    "TrimBGAdvanced": "TrimBG Advanced",
    "ColorImageFillRm": "Color Image Fill Rm",
    "ImageQueueLoader": "Image Queue Loader",
    "LoadImageA": "Load Image Alpha",
    "TxtSave": "Save Txt File",
    "TextSplitToList": "Text Split",
    "ScaleMultilplePixels": "Scale Image to Multilple Pixels",
    "FaceSimilarityTopK": "Face Similarity Check",
    "InsightFaceSimilarityTopK": "Face Similarity Check(Insight)",
    "HybridFaceSimilarityTopK": "Face Similarity Check(Hybrid)",
    "FaceIdentitySimilarity": "Face Identity Similarity (antelopev2)",
    "LoraLoad": "Lora Load",
    "LoraMerge": "Lora Merge",
    "LoraSaver": "Lora Save",
}

NODE_CLASS_MAPPINGS.update(XY_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(XY_NODE_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
