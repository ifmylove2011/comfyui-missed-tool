import logging
import os
import re

logger = logging.getLogger("image_queue")


class TxtSave:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "filename": ("STRING", {"default": '001.txt', "multiline": False}),
                "src_mode": ("BOOLEAN", {"default": True}),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "save_txt_file"
    CATEGORY = "missed-tool"

    def save_txt_file(self, text, filename, encoding='utf-8', src_mode=True):

        fname, fext = os.path.splitext(filename)
        if not src_mode:
            fname = filename
        dirname = os.path.dirname(filename)

        if not os.path.exists(dirname):
            logger.error(f"The path `{dirname}` doesn't exist! Creating it...")
            try:
                os.makedirs(dirname, exist_ok=True)
            except OSError as e:
                logger.error(f"The path `{dirname}` could not be created! Is there write access?\n{e}")

        if text.strip() == '':
            logger.error(f"There is no text specified to save! Text is empty.")

        self.write_text_file(f'{fname}.txt', text, encoding)
        return (text, {"ui": {"string": text}})

    def write_text_file(self, file, content, encoding):
        try:
            with open(file, 'w', encoding=encoding, newline='\n') as f:
                f.write(content)
        except OSError:
            logger.error(f"Unable to save file `{file}`")


class TextSplitToList:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "delimiter": ("STRING", {
                    "multiline": False,
                    "default": ",",
                }),
                "use_regex": (["disable", "enable"], {
                    "default": "disable",
                }),
                "strip_whitespace": (["disable", "enable"], {
                    "default": "enable",
                }),
                "skip_empty": (["disable", "enable"], {
                    "default": "enable",
                }),
                "replace_delimiter_with": ("STRING", {
                    "multiline": False,
                    "default": " ",  # 可以换成 "" 或 " "
                    "tooltip": "What to replace the delimiter with in the cleaned text (e.g., space, comma, nothing)."
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                }),
                "end_index": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": 9999,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("string_list", "body_text", "item_count")
    OUTPUT_IS_LIST = (True, False, False)  # 只有第一个是列表
    FUNCTION = "split_and_clean"
    CATEGORY = "missed-tool"

    def split_and_clean(self, input_text, delimiter, use_regex, strip_whitespace, skip_empty,
                       replace_delimiter_with, start_index, end_index):
        # Step 1: Split the text
        if use_regex == "enable":
            try:
                items = re.split(delimiter, input_text)
            except re.error as e:
                raise ValueError(f"Invalid regular expression: {e}")
        else:
            items = input_text.split(delimiter)

        # Step 2: Process items for list output
        processed_items = []
        for item in items:
            if strip_whitespace == "enable":
                item = item.strip()
            if skip_empty == "enable" and not item:
                continue
            processed_items.append(item)

        # Step 3: Slice the list based on start/end index
        total_items = len(processed_items)
        start_idx = max(0, min(start_index, total_items - 1)) if total_items > 0 else 0
        end_idx = max(start_idx, min(end_index, total_items))
        selected_items = processed_items[start_idx:end_idx]
        item_count = len(selected_items)

        # Step 4: Generate clean body_text (single string, no delimiters)
        # Option 1: Replace delimiter in original text
        if use_regex == "enable":
            # For regex, we use re.sub to remove/replace all matches
            try:
                cleaned_text = re.sub(delimiter, replace_delimiter_with, input_text)
            except re.error as e:
                cleaned_text = input_text  # fallback
        else:
            cleaned_text = input_text.replace(delimiter, replace_delimiter_with)

        # Optional: Normalize whitespace (e.g., "  " → " ")
        import re as regex
        cleaned_text = regex.sub(r'\s+', ' ', cleaned_text).strip()

        return (selected_items, cleaned_text, item_count)