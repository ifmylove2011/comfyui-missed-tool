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
    CATEGORY = "missing-tool"

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
