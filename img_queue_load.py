import json
import os
import glob
import logging
from .image_transf import *

NODE_NAME = 'ImageQueueLoader'
ALLOWED_EXT = ('.jpeg', '.jpg', '.png', '.bmp')
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, 'tool_data.json')

logger = logging.getLogger("image_queue")


class ImageQueueLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": '', "multiline": False}),
                "start": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "limit": ("INT", {"default": 10, "min": 1, "max": 9999, "step": 1}),
                "filename_reg": ("STRING", {"default": '*', "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "METADATA_RAW",)
    RETURN_NAMES = ("image", "Metadata RAW",)
    FUNCTION = "load_images"
    CATEGORY = "missed-tool"

    current_index = 0

    def load_images(self, path, start=0, limit=10, filename_reg='*'):
        image_paths = []

        if not os.path.exists(path):
            return None, None
        if os.path.isfile(path):
            path = os.path.dirname
        for file_name in glob.glob(os.path.join(glob.escape(path), filename_reg), recursive=False):
            if file_name.lower().endswith(ALLOWED_EXT):
                abs_file_path = os.path.abspath(file_name)
                image_paths.append(abs_file_path)

        if not os.path.exists(DATA_PATH):
            local_data = {"path": path, "start": start, "limit": limit, "filename_reg": filename_reg}
            self.set_local_data(local_data)

        return self.get_next_image(image_paths, start, limit)

    def get_next_image(self, image_paths, start, limit):
        local_data = self.get_local_data()

        current_index = local_data['start']
        if current_index >= min(start + limit, len(image_paths)):
            current_index = start
        image_path = image_paths[current_index]
        current_index += 1
        if current_index == min(start + limit, len(image_paths)):
            current_index = start
        local_data['start'] = current_index

        self.set_local_data(local_data)

        img = Image.open(image_path)
        return pil_to_tensor(img), self.get_metadata(img)

    def get_metadata(self, img):
        metadata = {}
        metadata_from_img = img.info

        # for all metadataFromImg convert to string (but not for workflow and prompt!)
        for k, v in metadata_from_img.items():
            # from ComfyUI
            if k == "workflow":
                try:
                    metadata["workflow"] = json.loads(metadata_from_img["workflow"])
                except Exception:
                    metadata["workflow"] = None

            # from ComfyUI
            elif k == "prompt":
                try:
                    metadata["prompt"] = json.loads(metadata_from_img["prompt"])
                except Exception:
                    metadata["prompt"] = None

            else:
                try:
                    metadata[str(k)] = json.loads(v)
                except Exception as e:
                    logger.debug(f"Error parsing {k} as json, trying as string: {e}")
                    try:
                        metadata[str(k)] = str(v)
                    except Exception as e:
                        logger.debug(f"Error parsing {k} it will be skipped: {e}")
        return metadata

    def get_local_data(self):
        with open(DATA_PATH, "r") as json_file:
            return json.load(json_file)

    def set_local_data(self, data):
        with open(DATA_PATH, "w") as json_file:
            json_file.write(json.dumps(data))

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
