import math
import os
import re

import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image, ImageDraw, ImageFont

import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import nodes


AXIS_TYPE = "MISSED_XY_AXIS"
AXIS_PARAMETERS = ["none", "seed", "steps", "cfg", "denoise", "sampler", "scheduler"]
SAMPLING_QUICK_PICKERS = 8


class AnyType(str):
    """ComfyUI wildcard socket that accepts primitive values and list outputs."""

    def __ne__(self, other):
        return False


ANY_TYPE = AnyType("*")


def _split_values(text, allow_commas=False):
    """Split on lines/semicolons while leaving commas available for file names."""
    separators = r"[,;\r\n]+" if allow_commas else r"[;\r\n]+"
    return [value.strip() for value in re.split(separators, str(text or "")) if value.strip()]


def _flatten_axis_values(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        return _flatten_axis_values(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _flatten_axis_values(value.tolist())
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_axis_values(item))
        return flattened
    if isinstance(value, str):
        return _split_values(value, allow_commas=True)
    return [value]


def _flatten_list_items(value):
    """Flatten connected list data while preserving each string verbatim."""
    if value is None:
        return []
    if torch.is_tensor(value):
        return _flatten_list_items(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _flatten_list_items(value.tolist())
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_list_items(item))
        return flattened
    return [value]


def _cast_axis_values(parameter, values):
    if parameter in ("seed", "steps"):
        result = [int(value) for value in values]
        if parameter == "seed" and any(value < 0 or value > 0xFFFFFFFFFFFFFFFF for value in result):
            raise ValueError("Seed must be between 0 and 18446744073709551615.")
        if parameter == "steps" and any(value < 1 or value > 10000 for value in result):
            raise ValueError("Steps must be between 1 and 10000.")
        return result
    if parameter in ("cfg", "denoise"):
        result = [float(value) for value in values]
        if parameter == "cfg" and any(value < 0 or value > 100 for value in result):
            raise ValueError("CFG must be between 0 and 100.")
        if parameter == "denoise" and any(value < 0 or value > 1 for value in result):
            raise ValueError("Denoise must be between 0 and 1.")
        return result
    if parameter == "sampler":
        invalid = [value for value in values if value not in comfy.samplers.KSampler.SAMPLERS]
        if invalid:
            raise ValueError(f"Unknown sampler: {invalid[0]}")
    if parameter == "scheduler":
        invalid = [value for value in values if value not in comfy.samplers.KSampler.SCHEDULERS]
        if invalid:
            raise ValueError(f"Unknown scheduler: {invalid[0]}")
    return values


class MissedXYAxis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "parameter": (AXIS_PARAMETERS,),
                "values": (
                    "STRING",
                    {
                        "default": "1, 2, 3",
                        "multiline": True,
                        "placeholder": "Example: 1, 2, 3  (new lines and semicolons also work)",
                        "tooltip": (
                            "Use new lines, commas, or semicolons between values. If values_list is connected, "
                            "the connected list takes priority. For sampler/scheduler, "
                            "prefer the dedicated dropdown-based XY Sampler Axis or XY Scheduler Axis node."
                        ),
                    },
                ),
            },
            "optional": {
                "values_list": (
                    ANY_TYPE,
                    {
                        "forceInput": True,
                        "tooltip": (
                            "Accepts STRING, INT, FLOAT, tensors/arrays, or ComfyUI list outputs. "
                            "Nested lists are flattened. This input overrides the values text box."
                        ),
                    },
                )
            },
        }

    RETURN_TYPES = (AXIS_TYPE,)
    RETURN_NAMES = ("axis",)
    FUNCTION = "make_axis"
    INPUT_IS_LIST = True
    CATEGORY = "missed tool/XY Plot"
    DESCRIPTION = (
        "Builds a seed, steps, CFG, denoise, sampler, or scheduler axis from text, comma-separated values, "
        "or a connected list."
    )

    def make_axis(self, parameter, values, values_list=None):
        parameter = parameter[0] if isinstance(parameter, (list, tuple)) else parameter
        if parameter == "none":
            return ({"parameter": "none", "values": [None], "labels": ["base"]},)
        source = values_list if values_list is not None else values
        parsed = _flatten_axis_values(source)
        if not parsed:
            raise ValueError(f"The {parameter} axis has no values.")
        parsed = _cast_axis_values(parameter, parsed)
        return ({"parameter": parameter, "values": parsed, "labels": [str(value) for value in parsed]},)


class _MissedXYChoiceAxis:
    PARAMETER = None
    ADDITIONAL_INPUT = None
    CHOICE_PREFIX = None
    CHOICES = None

    @classmethod
    def INPUT_TYPES(cls):
        options = ["None"] + list(cls.CHOICES())
        required = {
            cls.ADDITIONAL_INPUT: (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional additional exact names, one per line or separated with semicolons.",
                },
            )
        }
        for index in range(1, SAMPLING_QUICK_PICKERS + 1):
            required[f"{cls.CHOICE_PREFIX}_{index}"] = (options,)
        return {"required": required}

    def make_axis(self, **kwargs):
        available = list(self.CHOICES())
        available_set = set(available)
        values = []
        for index in range(1, SAMPLING_QUICK_PICKERS + 1):
            value = kwargs.get(f"{self.CHOICE_PREFIX}_{index}", "None")
            if value != "None" and value not in values:
                values.append(value)
        for value in _split_values(kwargs.get(self.ADDITIONAL_INPUT, "")):
            if value not in available_set:
                raise ValueError(f"Unknown {self.PARAMETER}: {value}")
            if value not in values:
                values.append(value)
        if not values:
            raise ValueError(f"The {self.PARAMETER} axis has no selected values.")
        return ({"parameter": self.PARAMETER, "values": values, "labels": values.copy()},)


class MissedXYSamplerAxis(_MissedXYChoiceAxis):
    PARAMETER = "sampler"
    ADDITIONAL_INPUT = "additional_samplers"
    CHOICE_PREFIX = "sampler"
    CHOICES = staticmethod(lambda: comfy.samplers.KSampler.SAMPLERS)
    RETURN_TYPES = (AXIS_TYPE,)
    RETURN_NAMES = ("axis",)
    FUNCTION = "make_axis"
    CATEGORY = "missed tool/XY Plot"
    DESCRIPTION = "Builds a sampler axis with dropdown selectors and an optional unlimited text list."


class MissedXYSchedulerAxis(_MissedXYChoiceAxis):
    PARAMETER = "scheduler"
    ADDITIONAL_INPUT = "additional_schedulers"
    CHOICE_PREFIX = "scheduler"
    CHOICES = staticmethod(lambda: comfy.samplers.KSampler.SCHEDULERS)
    RETURN_TYPES = (AXIS_TYPE,)
    RETURN_NAMES = ("axis",)
    FUNCTION = "make_axis"
    CATEGORY = "missed tool/XY Plot"
    DESCRIPTION = "Builds a scheduler axis with dropdown selectors and an optional unlimited text list."


class MissedXYLoraAxis:
    class _DynamicLoraInputs(dict):
        """Expose lora_2..lora_N to execution without bloating /object_info."""

        def __init__(self, input_spec):
            super().__init__()
            self.input_spec = input_spec

        @staticmethod
        def _is_dynamic_lora(name):
            if not isinstance(name, str) or not name.startswith("lora_"):
                return False
            suffix = name[5:]
            return suffix.isdigit() and int(suffix) >= 2

        def __contains__(self, key):
            return super().__contains__(key) or self._is_dynamic_lora(key)

        def __getitem__(self, key):
            if self._is_dynamic_lora(key):
                return self.input_spec
            return super().__getitem__(key)

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        required = {
            "compare_with_no_lora": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": (
                        "Add a 'No LoRA' coordinate that applies no LoRA from this axis. "
                        "LoRAs already applied upstream are not removed."
                    ),
                },
            ),
            "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "additional_loras": (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "placeholder": "One LoRA per line: folder\\name.safetensors|1.0",
                    "tooltip": (
                        "Optional unlimited list. Use one path per line; append |weight to override "
                        "the default model strength for that LoRA."
                    ),
                },
            ),
            "lora_count": (
                "INT",
                {
                    "default": 4,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Set the number of LoRA dropdowns, then click Update LoRA slots.",
                },
            ),
            "lora_1": (loras,),
        }
        return {
            "required": required,
            "optional": cls._DynamicLoraInputs((loras,)),
        }

    RETURN_TYPES = (AXIS_TYPE,)
    RETURN_NAMES = ("axis",)
    FUNCTION = "make_axis"
    SEARCH_ALIASES = ["xy axis lora", "xy lora", "lora axis", "lora comparison"]
    CATEGORY = "missed tool/XY Plot"
    DESCRIPTION = (
        "Builds an unlimited LoRA comparison axis. Set the dropdown count and click Update LoRA slots; "
        "additional LoRAs can also be entered one per line. LoRAs are applied to MODEL only."
    )

    @staticmethod
    def _parse_additional_loras(text, default_strength):
        available = set(folder_paths.get_filename_list("loras"))
        parsed = []
        for line in _split_values(text):
            name = line
            strength = default_strength
            if "|" in line:
                name, strength_text = line.rsplit("|", 1)
                name = name.strip()
                try:
                    strength = float(strength_text.strip())
                except ValueError as error:
                    raise ValueError(f"Invalid LoRA strength in line: {line}") from error
            if not name:
                continue
            if name not in available:
                raise ValueError(f"LoRA not found: {name}")
            if strength < -10.0 or strength > 10.0:
                raise ValueError(f"LoRA strength must be between -10 and 10: {line}")
            parsed.append({"name": name, "strength": strength})
        return parsed

    def make_axis(
        self,
        compare_with_no_lora,
        strength_model,
        additional_loras,
        lora_count=4,
        lora_1="None",
        **kwargs,
    ):
        values = []
        labels = []
        if compare_with_no_lora:
            values.append(None)
            labels.append("No LoRA")
        lora_count = max(1, min(100, int(lora_count)))
        selected_loras = {"lora_1": lora_1, **kwargs}
        for index in range(1, lora_count + 1):
            name = selected_loras.get(f"lora_{index}", "None")
            if name == "None":
                continue
            values.append({"name": name, "strength": float(strength_model)})
            short_name = os.path.splitext(os.path.basename(name))[0]
            labels.append(f"{short_name} (weight={strength_model:g})")
        for spec in self._parse_additional_loras(additional_loras, float(strength_model)):
            values.append(spec)
            short_name = os.path.splitext(os.path.basename(spec["name"]))[0]
            labels.append(f"{short_name} (weight={spec['strength']:g})")
        if not values:
            raise ValueError("The LoRA axis has no selected LoRAs. Select one or enable compare_with_no_lora.")
        return ({"parameter": "lora", "values": values, "labels": labels},)


class MissedXYPromptAxis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": (["positive", "negative"],),
                "prompts": (
                    "STRING",
                    {
                        "default": "first prompt\nsecond prompt\nthird prompt",
                        "multiline": True,
                        "placeholder": "One prompt per line, or connect prompts_list",
                        "tooltip": (
                            "One prompt per non-empty line. Commas and semicolons remain part of the prompt. "
                            "Use prompts_list when a prompt itself contains line breaks."
                        ),
                    },
                ),
                "label_mode": (["summary", "full prompt", "index"],),
            },
            "optional": {
                "prompts_list": (
                    ANY_TYPE,
                    {
                        "forceInput": True,
                        "tooltip": "Optional STRING list. It overrides the prompts text box and preserves each list item as one prompt.",
                    },
                )
            },
        }

    RETURN_TYPES = (AXIS_TYPE,)
    RETURN_NAMES = ("axis",)
    FUNCTION = "make_axis"
    INPUT_IS_LIST = True
    SEARCH_ALIASES = ["xy axis prompt", "prompt axis", "prompt comparison"]
    CATEGORY = "missed tool/XY Plot"
    DESCRIPTION = "Builds a positive or negative prompt axis. Connect CLIP to the main XY Plot node."

    def make_axis(self, target, prompts, label_mode, prompts_list=None):
        target = target[0] if isinstance(target, (list, tuple)) else target
        label_mode = label_mode[0] if isinstance(label_mode, (list, tuple)) else label_mode
        if prompts_list is not None:
            values = [str(item) for item in _flatten_list_items(prompts_list) if str(item).strip()]
        else:
            values = []
            for item in _flatten_list_items(prompts):
                values.extend(line.strip() for line in str(item).splitlines() if line.strip())
        if not values:
            raise ValueError("The prompt axis has no prompts.")
        if label_mode == "full prompt":
            labels = values.copy()
        elif label_mode == "index":
            labels = [f"Prompt {index + 1}" for index in range(len(values))]
        else:
            labels = [value if len(value) <= 80 else value[:79] + "…" for value in values]
        return ({"parameter": f"{target}_prompt", "values": values, "labels": labels},)


class MissedXYLoraWeightAxis:
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora_name": (loras,),
                "strength_values": (
                    "STRING",
                    {
                        "default": "0\n0.5\n0.8\n1.0\n1.2",
                        "multiline": True,
                        "tooltip": (
                            "One model strength per line (or separated with semicolons). Zero means no LoRA "
                            "from this axis; upstream LoRAs are not removed."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = (AXIS_TYPE,)
    RETURN_NAMES = ("axis",)
    FUNCTION = "make_axis"
    SEARCH_ALIASES = ["xy axis lora weight", "lora weight axis", "lora strength axis"]
    CATEGORY = "missed tool/XY Plot"
    DESCRIPTION = "Varies the model strength of one LoRA along an XY Plot axis."

    def make_axis(self, lora_name, strength_values):
        raw_values = _split_values(strength_values)
        if not raw_values:
            raise ValueError("The LoRA weight axis has no strength values.")
        try:
            strengths = [float(value) for value in raw_values]
        except ValueError as error:
            raise ValueError("Every LoRA strength must be a number.") from error
        invalid = next((value for value in strengths if value < -10.0 or value > 10.0), None)
        if invalid is not None:
            raise ValueError(f"LoRA strength must be between -10 and 10: {invalid}")
        short_name = os.path.splitext(os.path.basename(lora_name))[0]
        values = [None if strength == 0 else {"name": lora_name, "strength": strength} for strength in strengths]
        labels = [f"{short_name} (weight={strength:g})" for strength in strengths]
        return ({"parameter": "lora", "values": values, "labels": labels},)


class MissedXYPlot:
    def __init__(self):
        self._lora_cache = {}
        self._sample_cache = None
        self._decode_cache = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grid_spacing": ("INT", {"default": 8, "min": 0, "max": 256}),
                "label_font_size": ("INT", {"default": 56, "min": 10, "max": 100}),
                "show_axis_labels": ("BOOLEAN", {"default": True}),
                "left_label_width": (
                    "INT",
                    {
                        "default": 240,
                        "min": 0,
                        "max": 2048,
                        "tooltip": "Increase this to keep long wrapped Y-axis labels at the requested font size.",
                    },
                ),
                "top_label_height": (
                    "INT",
                    {
                        "default": 120,
                        "min": 0,
                        "max": 2048,
                        "tooltip": "Increase this to keep long wrapped X-axis labels at the requested font size.",
                    },
                ),
                "show_fixed_parameters": ("BOOLEAN", {"default": True}),
                "extra_info": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Model: model_name.safetensors; VAE: vae_name; Guidance: 3.0",
                        "tooltip": (
                            "Display-only information; it does not select or change the model. Add the model "
                            "name here if wanted. Separate items with semicolons, for example: "
                            "Model: z_image.safetensors; VAE: flux-ae.safetensors; Guidance: 3.0"
                        ),
                    },
                ),
            },
            "optional": {
                "x_axis": (AXIS_TYPE,),
                "y_axis": (AXIS_TYPE,),
                "clip": (
                    "CLIP",
                    {
                        "tooltip": (
                            "Optional. Connect CLIP only when an XY Prompt Axis is used; "
                            "ordinary seed/steps/CFG/LoRA plots do not need it."
                        ),
                    },
                ),
                "custom_sampler": ("SAMPLER", {"tooltip": "Optional: connect together with custom_sigmas when using a SamplerCustom-style workflow."}),
                "custom_sigmas": ("SIGMAS", {"tooltip": "Optional: connect together with custom_sampler when using a custom sigma schedule."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "LATENT", "STRING")
    RETURN_NAMES = ("grid", "individual_images", "samples", "plot_info")
    FUNCTION = "plot"
    CATEGORY = "missed tool/XY Plot"
    DESCRIPTION = (
        "Generates a labeled XY grid primarily through standard KSampler, or optional custom SAMPLER/SIGMAS. "
        "Works with local Z-Image and Krea 2 diffusion workflows; both can use standard KSampler mode."
    )

    @staticmethod
    def _axis_or_base(axis):
        return axis or {"parameter": "none", "values": [None], "labels": ["base"]}

    @classmethod
    def _freeze_value(cls, value):
        if isinstance(value, dict):
            return tuple(sorted((key, cls._freeze_value(item)) for key, item in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(cls._freeze_value(item) for item in value)
        if torch.is_tensor(value):
            # Tensors created inside torch.inference_mode() deliberately do not
            # expose a version counter. Accessing ``_version`` on them raises a
            # RuntimeError instead of returning a value.
            try:
                version = value._version
            except (AttributeError, RuntimeError):
                version = None
            return (
                "tensor",
                id(value),
                tuple(value.shape),
                str(value.dtype),
                str(value.device),
                version,
            )
        return value

    @classmethod
    def _sampling_cache_key(
        cls,
        model,
        positive,
        negative,
        clip,
        latent_image,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        x_axis,
        y_axis,
        using_custom,
        custom_sampler,
        custom_sigmas,
    ):
        varied = {x_axis["parameter"], y_axis["parameter"]}

        def fixed(name, value):
            return None if name in varied else value

        base_parameters = (
            fixed("seed", seed),
            fixed("steps", steps) if not using_custom or "steps" in varied else None,
            fixed("cfg", cfg),
            fixed("denoise", denoise),
            fixed("sampler", sampler_name) if not using_custom else None,
            fixed("scheduler", scheduler) if not using_custom else None,
        )
        latent_samples = latent_image.get("samples") if isinstance(latent_image, dict) else None
        return (
            "missed-xy-sampling-v2",
            id(model),
            id(positive) if "positive_prompt" not in varied else None,
            id(negative) if "negative_prompt" not in varied else None,
            id(clip) if varied & {"positive_prompt", "negative_prompt"} else None,
            id(latent_image),
            cls._freeze_value(latent_samples),
            base_parameters,
            (x_axis["parameter"], cls._freeze_value(x_axis["values"])),
            (y_axis["parameter"], cls._freeze_value(y_axis["values"])),
            using_custom,
            id(custom_sampler) if using_custom else None,
            cls._freeze_value(custom_sigmas) if using_custom else None,
        )

    @staticmethod
    def _set_parameter(parameters, axis, index):
        parameter = axis["parameter"]
        value = axis["values"][index]
        if parameter != "none":
            parameters[parameter] = value

    def _apply_lora(self, model, lora_spec):
        if not lora_spec:
            return model
        name = lora_spec["name"]
        path = folder_paths.get_full_path_or_raise("loras", name)
        cached = self._lora_cache.get(path)
        if cached is None:
            lora, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
            cached = (lora, metadata)
            self._lora_cache[path] = cached
        lora, metadata = cached
        patched_model, _ = comfy.sd.load_lora_for_models(
            model, None, lora, lora_spec["strength"], 0.0, lora_metadata=metadata
        )
        return patched_model

    @staticmethod
    def _custom_sigmas(sigmas, steps, denoise):
        if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or sigmas.numel() < 2:
            raise ValueError("custom_sigmas must be a one-dimensional SIGMAS tensor with at least two entries.")
        if denoise <= 0:
            return sigmas.new_zeros([1])
        full_steps = max(steps, int(math.ceil(steps / denoise)))
        if full_steps == sigmas.numel() - 1:
            resized = sigmas
        else:
            source = sigmas.to(dtype=torch.float32).reshape(1, 1, -1)
            resized = torch_functional.interpolate(source, size=full_steps + 1, mode="linear", align_corners=True)
            resized = resized.reshape(-1).to(device=sigmas.device, dtype=sigmas.dtype)
        return resized[-(steps + 1):]

    @staticmethod
    def _sample_custom(model, seed, cfg, positive, negative, latent, sampler, sigmas):
        latent_out = latent.copy()
        latent_samples = comfy.sample.fix_empty_latent_channels(
            model,
            latent["samples"],
            latent.get("downscale_ratio_spacial"),
            latent.get("downscale_ratio_temporal"),
        )
        batch_indices = latent.get("batch_index")
        noise = comfy.sample.prepare_noise(latent_samples, seed, batch_indices)
        samples = comfy.sample.sample_custom(
            model,
            noise,
            cfg,
            sampler,
            sigmas,
            positive,
            negative,
            latent_samples,
            noise_mask=latent.get("noise_mask"),
            disable_pbar=True,
            seed=seed,
        )
        latent_out.pop("downscale_ratio_spacial", None)
        latent_out.pop("downscale_ratio_temporal", None)
        latent_out["samples"] = samples
        return latent_out

    @staticmethod
    def _load_font(size):
        candidates = [
            "C:/Windows/Fonts/msyh.ttc",
            "arial.ttf",
            "DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
        return ImageFont.load_default()

    @staticmethod
    def _normalize_image_batch(images):
        """Normalize VAE outputs (including BTHWC video-style outputs) to BHWC."""
        if not torch.is_tensor(images):
            raise TypeError(f"Expected an IMAGE tensor, got {type(images).__name__}.")
        if images.ndim == 3:
            images = images.unsqueeze(0)
        elif images.ndim > 4:
            # Some image VAEs retain one or more singleton/frame dimensions.
            # Treat every leading dimension as part of the image batch.
            images = images.reshape(-1, *images.shape[-3:])
        if images.ndim != 4:
            raise ValueError(f"IMAGE must resolve to BHWC, received shape {tuple(images.shape)}.")
        # Also accept BCHW defensively, though ComfyUI normally uses BHWC.
        if images.shape[-1] not in (1, 3, 4) and images.shape[1] in (1, 3, 4):
            images = images.permute(0, 2, 3, 1)
        if images.shape[-1] == 1:
            images = images.repeat(1, 1, 1, 3)
        if images.shape[-1] not in (3, 4):
            raise ValueError(f"IMAGE must have 1, 3, or 4 channels, received shape {tuple(images.shape)}.")
        return images

    @classmethod
    def _tensor_to_pil(cls, image):
        # Accept either a single HWC image or a batch/frame-wrapped image.
        if image.ndim != 3:
            image = cls._normalize_image_batch(image)[0]
        if image.shape[-1] == 1:
            image = image.repeat(1, 1, 3)
        array = (image.detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(array)

    @classmethod
    def _make_grid(
        cls,
        cell_images,
        x_axis,
        y_axis,
        spacing,
        font_size,
        common_info="",
        left_label_width=240,
        top_label_height=120,
        show_axis_labels=True,
    ):
        columns = len(x_axis["values"])
        rows = len(y_axis["values"])
        first = cell_images[0]
        cell_width, cell_height = first.size
        top = top_label_height if show_axis_labels and x_axis["parameter"] != "none" else 0
        left = left_label_width if show_axis_labels and y_axis["parameter"] != "none" else 0
        width = left + columns * cell_width + max(0, columns - 1) * spacing
        info_lines, info_font = cls._layout_common_info(common_info, width, font_size)
        info_height = (font_size + 8) * len(info_lines) + 12 if info_lines else 0
        grid_body_height = top + rows * cell_height + max(0, rows - 1) * spacing
        height = grid_body_height + info_height
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)

        for row in range(rows):
            for column in range(columns):
                image = cell_images[row * columns + column]
                if image.size != (cell_width, cell_height):
                    image = image.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
                x = left + column * (cell_width + spacing)
                y = top + row * (cell_height + spacing)
                canvas.paste(image.convert("RGB"), (x, y))

        if top:
            for column, label in enumerate(x_axis["labels"]):
                x = left + column * (cell_width + spacing)
                box = (x, 0, x + cell_width, top)
                cls._draw_centered(draw, str(label), box, font_size)
        if top and left:
            cls._draw_centered(
                draw,
                f"{y_axis['parameter']} \\ {x_axis['parameter']}",
                (0, 0, left, top),
                font_size,
            )
        if left:
            for row, label in enumerate(y_axis["labels"]):
                y = top + row * (cell_height + spacing)
                # Rotate Y labels so long LoRA/model names retain a useful font size.
                panel = Image.new("RGB", (cell_height, left), "white")
                panel_draw = ImageDraw.Draw(panel)
                cls._draw_centered(panel_draw, str(label), (0, 0, cell_height, left), font_size)
                canvas.paste(panel.rotate(90, expand=True), (0, y))
        if info_lines:
            cls._draw_common_info(
                draw, info_lines, (0, grid_body_height, width, height), info_font, font_size + 8
            )
        return torch.from_numpy(np.asarray(canvas).copy()).float().div_(255.0).unsqueeze(0)

    @classmethod
    def _draw_centered(cls, draw, text, box, font_size):
        available_width = max(1, box[2] - box[0] - 8)
        available_height = max(1, box[3] - box[1] - 6)
        current_size = font_size
        wrapped_text = text
        while True:
            font = cls._load_font(current_size)
            spacing = max(2, current_size // 5)
            wrapped_text = cls._wrap_label_text(draw, text, font, available_width)
            bounds = draw.multiline_textbbox(
                (0, 0), wrapped_text, font=font, spacing=spacing, align="center"
            )
            if bounds[2] - bounds[0] <= available_width and bounds[3] - bounds[1] <= available_height:
                break
            if current_size <= 8:
                break
            current_size -= 2
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        x = box[0] + (box[2] - box[0] - text_width) / 2
        y = box[1] + (box[3] - box[1] - text_height) / 2 - bounds[1]
        draw.multiline_text(
            (x, y), wrapped_text, fill="black", font=font, spacing=spacing, align="center"
        )

    @staticmethod
    def _wrap_label_text(draw, text, font, max_width):
        """Pixel-wrap labels without dropping characters, including long filenames with no spaces."""
        lines = []
        for paragraph in str(text).splitlines() or [""]:
            if not paragraph:
                lines.append("")
                continue
            current = ""
            for character in paragraph:
                candidate = current + character
                if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
                    break_at = max((current.rfind(separator) for separator in " _-/\\"), default=-1)
                    if break_at >= max(1, len(current) // 3):
                        lines.append(current[: break_at + 1].rstrip())
                        current = current[break_at + 1 :].lstrip() + character
                    else:
                        lines.append(current)
                        current = character
                else:
                    current = candidate
            if current:
                lines.append(current)
        return "\n".join(lines)

    @classmethod
    def _layout_common_info(cls, common_info, width, font_size):
        raw_items = [item.strip() for item in re.split(r"[;；\r\n]+", common_info or "") if item.strip()]
        items = []
        for item in raw_items:
            match = re.match(r"^\s*([^:=：]{1,48}?)\s*[:=：]\s*(.*?)\s*$", item)
            if match and match.group(2):
                items.append(f"{match.group(1).strip()}: {match.group(2).strip()}")
            else:
                items.append(item)
        if not items:
            return [], cls._load_font(font_size)

        measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        available_width = max(1, width - 24)
        current_size = font_size
        font = cls._load_font(current_size)
        while current_size > 8 and any(
            measure.textbbox((0, 0), item, font=font)[2] > available_width for item in items
        ):
            current_size -= 2
            font = cls._load_font(current_size)

        lines = []
        current_line = ""
        for item in items:
            candidate = f"{current_line}; {item}" if current_line else item
            if current_line and measure.textbbox((0, 0), candidate, font=font)[2] > available_width:
                lines.append(current_line)
                current_line = item
            else:
                current_line = candidate
        if current_line:
            lines.append(current_line)
        return lines, font

    @staticmethod
    def _draw_common_info(draw, lines, box, font, row_height):
        draw.rectangle(box, fill=(246, 246, 246))
        draw.line((box[0], box[1], box[2], box[1]), fill=(210, 210, 210), width=1)
        for index, line in enumerate(lines):
            y = box[1] + 6 + index * row_height
            draw.text((box[0] + 12, y), line, fill="black", font=font)

    @staticmethod
    def _build_common_info(
        extra_info,
        show_fixed_parameters,
        x_axis,
        y_axis,
        using_custom,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        resolution,
    ):
        lines = []
        varied = {x_axis["parameter"], y_axis["parameter"]}
        if show_fixed_parameters:
            candidates = [
                ("resolution", resolution),
                ("seed", seed),
                ("steps", steps),
                ("cfg", f"{cfg:g}"),
                ("denoise", f"{denoise:g}"),
            ]
            if not using_custom:
                candidates.extend((("sampler", sampler_name), ("scheduler", scheduler)))
            mode = "custom SAMPLER/SIGMAS" if using_custom else "KSampler"
            lines.append(f"Mode: {mode}")
            for name, value in candidates:
                if name not in varied:
                    lines.append(f"{name}: {value}")
        lines.extend(line.strip() for line in (extra_info or "").splitlines() if line.strip())
        return "; ".join(lines)

    @staticmethod
    def _merge_latents(latents):
        result = latents[0].copy()
        result["samples"] = torch.cat([latent["samples"] for latent in latents], dim=0)
        result.pop("batch_index", None)
        result.pop("noise_mask", None)
        return result

    def plot(
        self,
        model,
        positive,
        negative,
        latent_image,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        grid_spacing,
        label_font_size,
        show_axis_labels,
        left_label_width,
        top_label_height,
        show_fixed_parameters,
        extra_info,
        x_axis=None,
        y_axis=None,
        clip=None,
        custom_sampler=None,
        custom_sigmas=None,
    ):
        x_axis = self._axis_or_base(x_axis)
        y_axis = self._axis_or_base(y_axis)
        if x_axis["parameter"] != "none" and x_axis["parameter"] == y_axis["parameter"]:
            raise ValueError("X and Y axes cannot control the same parameter.")
        using_custom = custom_sampler is not None or custom_sigmas is not None
        if using_custom and (custom_sampler is None or custom_sigmas is None):
            raise ValueError("custom_sampler and custom_sigmas must be connected together.")
        if using_custom and ({x_axis["parameter"], y_axis["parameter"]} & {"sampler", "scheduler"}):
            raise ValueError("Sampler and scheduler axes are only available in standard KSampler mode.")
        prompt_axes = {x_axis["parameter"], y_axis["parameter"]} & {"positive_prompt", "negative_prompt"}
        if prompt_axes and clip is None:
            raise ValueError("Connect CLIP to the XY Plot node when using an XY Prompt Axis.")
        custom_steps_axis = "steps" in {x_axis["parameter"], y_axis["parameter"]}

        total = len(x_axis["values"]) * len(y_axis["values"])
        sample_key = self._sampling_cache_key(
            model,
            positive,
            negative,
            clip,
            latent_image,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            x_axis,
            y_axis,
            using_custom,
            custom_sampler,
            custom_sigmas,
        )
        if self._sample_cache is not None and self._sample_cache[0] == sample_key:
            sampled_latents, merged_latent = self._sample_cache[1:]
        else:
            progress = comfy.utils.ProgressBar(total)
            sampled_latents = []
            for y_index in range(len(y_axis["values"])):
                for x_index in range(len(x_axis["values"])):
                    parameters = {
                        "seed": seed,
                        # In custom mode the connected SIGMAS owns the base step count.
                        # The widget becomes active again when an axis explicitly varies steps.
                        "steps": (custom_sigmas.numel() - 1) if using_custom and not custom_steps_axis else steps,
                        "cfg": cfg,
                        "sampler": sampler_name,
                        "scheduler": scheduler,
                        "denoise": denoise,
                        "lora": None,
                        "positive_prompt": None,
                        "negative_prompt": None,
                    }
                    self._set_parameter(parameters, x_axis, x_index)
                    self._set_parameter(parameters, y_axis, y_index)
                    cell_model = self._apply_lora(model, parameters["lora"])
                    cell_positive = positive
                    cell_negative = negative
                    if parameters["positive_prompt"] is not None:
                        cell_positive = nodes.CLIPTextEncode().encode(
                            clip, parameters["positive_prompt"]
                        )[0]
                    if parameters["negative_prompt"] is not None:
                        cell_negative = nodes.CLIPTextEncode().encode(
                            clip, parameters["negative_prompt"]
                        )[0]

                    if using_custom:
                        cell_sigmas = self._custom_sigmas(
                            custom_sigmas, parameters["steps"], parameters["denoise"]
                        )
                        sampled = self._sample_custom(
                            cell_model,
                            parameters["seed"],
                            parameters["cfg"],
                            cell_positive,
                            cell_negative,
                            latent_image,
                            custom_sampler,
                            cell_sigmas,
                        )
                    else:
                        sampled = nodes.common_ksampler(
                            cell_model,
                            parameters["seed"],
                            parameters["steps"],
                            parameters["cfg"],
                            parameters["sampler"],
                            parameters["scheduler"],
                            cell_positive,
                            cell_negative,
                            latent_image,
                            denoise=parameters["denoise"],
                        )[0]
                    sampled_latents.append(sampled)
                    progress.update(1)
            merged_latent = self._merge_latents(sampled_latents)
            self._sample_cache = (sample_key, sampled_latents, merged_latent)
            self._decode_cache = None
            self._lora_cache.clear()

        decode_key = (sample_key, id(vae))
        if self._decode_cache is not None and self._decode_cache[0] == decode_key:
            individual_images, grid_cells = self._decode_cache[1:]
        else:
            decoded_batches = []
            grid_cells = []
            for sampled in sampled_latents:
                images = self._normalize_image_batch(vae.decode(sampled["samples"]))
                decoded_batches.append(images)
                grid_cells.append(self._tensor_to_pil(images[0]))
            individual_images = torch.cat(decoded_batches, dim=0)
            self._decode_cache = (decode_key, individual_images, grid_cells)

        displayed_steps = custom_sigmas.numel() - 1 if using_custom and not custom_steps_axis else steps
        common_info = self._build_common_info(
            extra_info,
            show_fixed_parameters,
            x_axis,
            y_axis,
            using_custom,
            seed,
            displayed_steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            f"{grid_cells[0].width} x {grid_cells[0].height}",
        )
        grid = self._make_grid(
            grid_cells,
            x_axis,
            y_axis,
            grid_spacing,
            label_font_size,
            common_info=common_info,
            left_label_width=left_label_width,
            top_label_height=top_label_height,
            show_axis_labels=show_axis_labels,
        )
        # This output can be connected directly to the secondary grid composer.
        info = common_info
        return grid, individual_images, merged_latent, info


class MissedXYPlotGrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Usually individual_images from the XY Plot node."}),
                "grid_spacing": ("INT", {"default": 8, "min": 0, "max": 256}),
                "footer_spacing": ("INT", {"default": 0, "min": 0, "max": 256}),
                "label_font_size": ("INT", {"default": 56, "min": 10, "max": 100}),
                "show_axis_labels": ("BOOLEAN", {"default": True}),
                "left_label_width": (
                    "INT",
                    {
                        "default": 240,
                        "min": 0,
                        "max": 2048,
                        "tooltip": "Increase this to keep long wrapped Y-axis labels at the requested font size.",
                    },
                ),
                "top_label_height": (
                    "INT",
                    {
                        "default": 120,
                        "min": 0,
                        "max": 2048,
                        "tooltip": "Increase this to keep long wrapped X-axis labels at the requested font size.",
                    },
                ),
            },
            "optional": {
                "x_axis": (AXIS_TYPE,),
                "y_axis": (AXIS_TYPE,),
                "footer_images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional footer appended below each generated image, such as a rendered score. "
                            "Use one image to repeat it, or one image per XY cell."
                        )
                    },
                ),
                "common_info": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect plot_info to preserve the model/fixed-parameter panel in the rebuilt grid.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("grid", "cells_with_footer")
    FUNCTION = "compose"
    CATEGORY = "missed tool/XY Plot"
    DESCRIPTION = (
        "Appends an optional footer IMAGE below each XY cell image, then rebuilds the grid. "
        "This allows score/evaluation nodes to run between generation and final grid composition."
    )

    @staticmethod
    def _select_cell_images(images, cell_count):
        image_count = images.shape[0]
        if image_count < cell_count or image_count % cell_count != 0:
            raise ValueError(
                f"Image batch ({image_count}) must contain the same number of images as XY cells "
                f"({cell_count}), or an equal-size batch for every cell."
            )
        batch_per_cell = image_count // cell_count
        return [images[index * batch_per_cell] for index in range(cell_count)], batch_per_cell

    @staticmethod
    def _select_footer_images(footer_images, cell_count, image_count, batch_per_cell):
        extra_count = footer_images.shape[0]
        if extra_count == 1:
            return [footer_images[0]] * cell_count
        if extra_count == cell_count:
            return [footer_images[index] for index in range(cell_count)]
        if extra_count == image_count:
            return [footer_images[index * batch_per_cell] for index in range(cell_count)]
        raise ValueError(
            f"Footer image batch ({extra_count}) must be 1, the XY cell count ({cell_count}), "
            f"or the source image batch count ({image_count})."
        )

    @staticmethod
    def _append_footer(image, footer, spacing):
        image = image.convert("RGB")
        footer = footer.convert("RGB")
        if footer.width != image.width:
            target_height = max(1, round(footer.height * image.width / footer.width))
            footer = footer.resize((image.width, target_height), Image.Resampling.LANCZOS)
        result = Image.new("RGB", (image.width, image.height + spacing + footer.height), "white")
        result.paste(image, (0, 0))
        result.paste(footer, (0, image.height + spacing))
        return result

    def compose(
        self,
        images,
        grid_spacing,
        footer_spacing,
        label_font_size,
        show_axis_labels,
        left_label_width,
        top_label_height,
        x_axis=None,
        y_axis=None,
        footer_images=None,
        common_info="",
    ):
        images = MissedXYPlot._normalize_image_batch(images)
        if footer_images is not None:
            footer_images = MissedXYPlot._normalize_image_batch(footer_images)
        x_axis = MissedXYPlot._axis_or_base(x_axis)
        y_axis = MissedXYPlot._axis_or_base(y_axis)
        cell_count = len(x_axis["values"]) * len(y_axis["values"])
        selected, batch_per_cell = self._select_cell_images(images, cell_count)
        cells = [MissedXYPlot._tensor_to_pil(image) for image in selected]

        if footer_images is not None:
            footers = self._select_footer_images(footer_images, cell_count, images.shape[0], batch_per_cell)
            cells = [
                self._append_footer(cell, MissedXYPlot._tensor_to_pil(footer), footer_spacing)
                for cell, footer in zip(cells, footers)
            ]

        grid = MissedXYPlot._make_grid(
            cells,
            x_axis,
            y_axis,
            grid_spacing,
            label_font_size,
            common_info=common_info,
            left_label_width=left_label_width,
            top_label_height=top_label_height,
            show_axis_labels=show_axis_labels,
        )
        cell_tensors = [
            torch.from_numpy(np.asarray(cell).copy()).float().div_(255.0).unsqueeze(0) for cell in cells
        ]
        return grid, torch.cat(cell_tensors, dim=0)


NODE_CLASS_MAPPINGS = {
    "MissedXYAxis": MissedXYAxis,
    "MissedXYLoraAxis": MissedXYLoraAxis,
    "MissedXYLoraWeightAxis": MissedXYLoraWeightAxis,
    "MissedXYPromptAxis": MissedXYPromptAxis,
    "MissedXYSamplerAxis": MissedXYSamplerAxis,
    "MissedXYSchedulerAxis": MissedXYSchedulerAxis,
    "MissedXYPlot": MissedXYPlot,
    "MissedXYPlotGrid": MissedXYPlotGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MissedXYAxis": "XY Axis (Missed Tool)",
    "MissedXYLoraAxis": "XY LoRA Axis (Missed Tool)",
    "MissedXYLoraWeightAxis": "XY LoRA Weight Axis (Missed Tool)",
    "MissedXYPromptAxis": "XY Prompt Axis (Missed Tool)",
    "MissedXYSamplerAxis": "XY Sampler Axis (Missed Tool)",
    "MissedXYSchedulerAxis": "XY Scheduler Axis (Missed Tool)",
    "MissedXYPlot": "XY Plot KSampler - Z-Image / Krea 2",
    "MissedXYPlotGrid": "XY Cell Footer Composer",
}
