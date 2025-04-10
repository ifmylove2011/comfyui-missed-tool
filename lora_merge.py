import os
import torch
from safetensors.torch import load_file, save_file
import folder_paths


def pad_tensors(t1, t2):
    max_size = [max(s1, s2) for s1, s2 in zip(t1.size(), t2.size())]
    pt1 = torch.zeros(max_size, device=t1.device, dtype=t1.dtype)
    pt2 = torch.zeros(max_size, device=t2.device, dtype=t2.dtype)
    pt1[tuple(slice(0, s) for s in t1.size())] = t1
    pt2[tuple(slice(0, s) for s in t2.size())] = t2
    return pt1, pt2


class LoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_file": (folder_paths.get_filename_list("loras"),),
            }
        }

    RETURN_TYPES = ("LORA_TENSOR_DICT",)
    RETURN_NAMES = ("lora_model",)
    FUNCTION = "load"
    CATEGORY = "missed-tool"

    def load(self, lora_file):
        lora_path = os.path.join(folder_paths.models_dir, "loras", lora_file)
        lora_model = load_file(lora_path)
        return (lora_model,)


class LoraMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "main_lora": ("LORA_TENSOR_DICT",),
                "merge_lora": ("LORA_TENSOR_DICT",),
                "main_weight": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("LORA_TENSOR_DICT",)
    RETURN_NAMES = ("lora_model",)
    FUNCTION = "merge"
    CATEGORY = "missed-tool"

    def merge(self, main_lora, merge_lora, main_weight):
        main_weight = main_weight / 100.0
        merged = {}
        all_keys = set(main_lora.keys()).union(set(merge_lora.keys()))

        for key in all_keys:
            t1 = main_lora.get(key)
            t2 = merge_lora.get(key)

            if t1 is not None and t2 is not None:
                if t1.size() != t2.size():
                    t1, t2 = pad_tensors(t1, t2)
                merged[key] = main_weight * t1 + (1 - main_weight) * t2
            elif t1 is not None:
                merged[key] = t1
            else:
                merged[key] = t2

        return (merged,)


class LoraSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_model": ("LORA_TENSOR_DICT",),
                "savename": ("STRING", {"default": "merged_lora"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "missed-tool"
    OUTPUT_NODE = True

    def save(self, lora_model, savename):
        save_path = os.path.join(folder_paths.models_dir, "loras", f"{savename}.safetensors")
        save_file(lora_model, save_path)
        print(f"✅ LoRA 已保存到: {save_path}")
        return (save_path,)
