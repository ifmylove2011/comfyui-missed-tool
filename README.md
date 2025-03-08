# comfyui-missing-tool
A few tools for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), perhaps it's exactly what you need.

## Installation

To install comfyui-missing-tool, clone the repository and install the dependencies:

```bash
git clone https://github.com/ifmylove2011/comfyui-missing-tool.git
cd comfyui-missing-tool
pip install -r requirements.txt
```

## Table of contents
- [Image Queue Loader](#ImageQueueLoader)
- [Load Image alpha](#LoadImagealpha)
- [TrimBG](#TrimBG)


### ImageQueueLoader
Load Images in Queue, it is possible to obtain metadata information, such as seed, from the generated images in a queue. 
We can retrieve seeds sequentially during sequential execution.
![Image Queue Loader](./assets/loadqueue.png)
### LoadImagealpha
Load Image with alpha, it works when transparent images exits. Built-in load-image extension output is "RGB" default, not "RGBA".
![Load alpha](./assets/loadtrim.png)
Of course, we can also use the built-in load image of Comfyui, but it needs to be used in conjunction with a mask.
![Load alpha1](./assets/loadtrim1.png)
### TrimBG
We can also batch read images and process them. Better with [Load Image List From Dir](https://github.com/ltdrdata/ComfyUI-Inspire-Pack?tab=readme-ov-file#image-util), list, not batch.
![TrimBG1](./assets/trimbg1.png)
And, Better to use with [ComfyUI_BiRefNet_ll](https://github.com/lldacing/ComfyUI_BiRefNet_ll)
![TrimBG2](./assets/load_biref_trim.png)

## Contributing

If you want to contribute to comfyui-missing-tool, please fork the repository and create a pull request with your changes.

## License

comfyui-missing-tool is licensed under the MIT License. See the LICENSE file for more details.

## Contact

For any questions or feedback, please open an issue on GitHub or contact the repository owner.

Enjoy.
