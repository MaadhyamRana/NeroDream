# DeepDream prototype (Level 1: layer/channel activation maximization)

Laptop-only, Linux-developed, no UI yet -- CLI in, PNG out. Built so the
octave/gradient-ascent loop (`deepdream/core.py`) has zero dependency on
torch or any specific model, which is what makes it unit-testable and what
will make Level 2 (guided dreaming, two images) a small addition rather than
a rewrite.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The first time you run the CLI with a given `--model`, torchvision downloads
its pretrained ImageNet weights automatically (needs internet, one-time per
model) and caches them to `~/.cache/torch/hub/checkpoints/`. No manual
download step is needed.

## Run the tests

```bash
pytest tests/ -v
```

These 14 tests only touch `image_utils.py` and `core.py`, which don't import
torch at all -- they run in well under a second and don't need any model
downloaded. `models.py` (the torch-dependent part) isn't unit tested here on
purpose; it's a thin wrapper around torchvision, and the real test of it is
just running the CLI and looking at the output image.

## Run the dreamer

```bash
python cli.py --image your_photo.jpg --model mobilenet_v2
```

See every hookable layer name for a model (useful for finding your own
favorite layer to maximize):

```bash
python cli.py --model vgg16 --list-layers
```

Then try, e.g.:

```bash
python cli.py --image your_photo.jpg --model vgg16 --layer features.17 --octaves 5 --iterations 20
```

## Where to get more models (for your swap-and-test directory)

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

## Roadmap hook for Level 2

`core.run_octaves` takes a `compute_grad_fn: (image) -> gradient` and knows
nothing about how that gradient is produced. `models.make_torch_grad_fn` is
today's implementation ("maximize this layer's activation"). Level 2 (guided
dreaming with a second reference image) is a new function with the same
signature -- e.g. `make_guided_grad_fn(model, layer, guide_image)` that
minimizes distance to the guide's activation instead -- dropped in as a
straight swap in `cli.py`. No change needed to `core.py` or `image_utils.py`.
