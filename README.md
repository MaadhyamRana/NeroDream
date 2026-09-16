# NeroDream: DeepDream remix 

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
The first time you run the CLI with a given `--model`, torchvision downloads
its pretrained ImageNet weights automatically (needs internet, one-time per
model) and caches them to `~/.cache/torch/hub/checkpoints/`.


## Run the tests

```bash
pytest tests/ -v
```
These 14 tests only touch `image_utils.py` and `core.py`, which don't import
torch at all -- they run in well under a second and don't need any model
downloaded. `models.py` is 'tested' by looking at the output image.


## Run the dreamer

Basic run (default model is mobilenet_v2)
```bash
python cli.py --image your_photo.jpg
```
See every hookable layer name for a model (useful for finding your own
favorite layer to maximize):
```bash
python cli.py --model vgg16 --list-layers
```
More params; Custom run:
```bash
python cli.py --image your_photo.jpg --model vgg16 --layer features.17 --octaves 5 --iterations 20
```


## Where to get more models

Every model below is already wired up by name in `deepdream/models.py` --
just pass `--model <name>`. To add more later, these are the libraries to
pull from:

- **torchvision** (what this project uses): https://pytorch.org/vision/stable/models.html -- the full list of pretrained classifiers, all auto-downloading via the same `weights=...Weights.DEFAULT` pattern used in `models.py`.
- **Keras Applications** (if you ever port to TensorFlow/TF.js for the browser phase): https://keras.io/api/applications/ -- same set of architectures (InceptionV3, VGG, MobileNet, EfficientNet, ResNet, Xception), auto-downloads to `~/.keras/`.
- **timm** (PyTorch Image Models, by Ross Wightman): https://github.com/huggingface/pytorch-image-models -- 900+ architectures if you want more variety than the eight below.
- **ONNX Model Zoo**: https://github.com/onnx/models -- pretrained `.onnx` files directly, relevant later if you target ONNX Runtime Web for a browser build.

Currently registered (`--model` value, approx. download size, character):

| name | size | note |
|---|---|---|
| `mobilenet_v2` | 14 MB | fastest, most forgiving of small octaves -- the default |
| `mobilenet_v3_large` | 22 MB | similar to v2, slightly different bias |
| `efficientnet_b0` | 20 MB | good quality-per-MB, less battle-tested for dreaming |
| `googlenet` | 50 MB | same lineage as the original DeepDream network |
| `resnet50` | 98 MB | residual connections, more abstract look |
| `inception_v3` | 104 MB | the original DeepDream backbone, classic look |
| `vgg16` | 528 MB | clean conv stack, also the standard style-transfer backbone |
| `vgg19` | 548 MB | deeper VGG, richer textures, biggest download |

## Image size notes

There's no hard requirement. Guidance:

- `--max-size` (default 768px longest side) controls speed -- on CPU-only,
  a few hundred px keeps a full run to seconds-to-low-minutes; push it up if
  you have a CUDA GPU (`torch.cuda.is_available()`).
- The **smallest octave** matters more than the input size: some models
  (inception_v3, googlenet, resnet50) start failing or degrading once their
  smallest octave drops below ~64-75px, because internal pooling can shrink
  a feature map to nothing. `mobilenet_v2`/`v3`/`vgg16`/`vgg19` tolerate much
  smaller. The CLI prints a warning if your settings would go below a
  model's recommended minimum -- fix it by lowering `--octaves` or
  `--octave-scale`.

## Planned changes

Guided Dreaming with a second reference image - new function with the same
signature -- e.g. `make_guided_grad_fn(model, layer, guide_image)` that
minimizes distance to the guide's activation instead -- dropped in as a
straight swap in `cli.py`. No change needed to `core.py` or `image_utils.py`.

UI addition and simplification (native app / web app?)

