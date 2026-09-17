"""Simple DeepDream CLI: python cli.py --image photo.jpg

Needs torch + torchvision installed (see requirements.txt) and, on first use
of a given --model, an internet connection to download its pretrained
weights (cached afterwards to ~/.cache/torch/hub/checkpoints/).
"""
from __future__ import annotations

import argparse

from deepdream.core import run_octaves
from deepdream.image_utils import build_octave_sizes, load_image, save_image
from deepdream.models import AVAILABLE_MODELS, get_model, list_layers, make_guided_grad_fn, make_torch_grad_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple DeepDream: layer/channel activation maximization.")
    parser.add_argument("--image", help="Path to input photo.")
    parser.add_argument("--output", default="dream.jpg", help="Where to save the result.")
    parser.add_argument("--model", default="inception_v3", choices=list(AVAILABLE_MODELS))
    parser.add_argument("--layer", default="", help="Layer to maximize. Defaults to a good known layer per model.")
    parser.add_argument("--channel", type=int, default=-1, help="Maximize one channel instead of the whole layer.")
    parser.add_argument("--guide", default=None, help="Path to a guide image. If given, pulls the source's activation toward the guide's instead of maximizing it (ignores --channel).")
    parser.add_argument("--octaves", type=int, default=4)
    parser.add_argument("--octave-scale", type=float, default=1.8)
    parser.add_argument("--iterations", type=int, default=15, help="Gradient ascent steps per octave.")
    parser.add_argument("--step-size", type=float, default=0.005)
    parser.add_argument("--jitter", type=int, default=16, help="Max pixel jitter shift per step, 0 to disable.")
    parser.add_argument("--max-size", type=int, default=768, help="Downscale input so its longest side is at most this many pixels.")
    parser.add_argument("--list-layers", action="store_true", help="Print every hookable layer name for --model and exit.")
    args = parser.parse_args()

    model = get_model(args.model)

    if args.list_layers:
        for name in list_layers(model):
            print(name)
        return

    if not args.image:
        parser.error("--image is required unless --list-layers is given")

    # fetching the layer/ default from the model
    layer = AVAILABLE_MODELS[args.model].default_layer if args.layer == "" else args.layer
    min_px = AVAILABLE_MODELS[args.model].min_recommended_input_px

    # prepare the gradient ascent function from the selected layer
    if args.guide:
        guide_image = load_image(args.guide, max_size=args.max_size)
        grad_fn = make_guided_grad_fn(model, layer, guide_image)
    else:
        grad_fn = make_torch_grad_fn(model, layer, args.channel)

    # runitt
    original = load_image(args.image, max_size=args.max_size)
    sizes = build_octave_sizes(original.shape[:2], args.octaves, args.octave_scale)

    if min(sizes[0]) < min_px:
        print(
            f"Warning: smallest octave {sizes[0]} is below the recommended "
            f"{min_px}px for {args.model} -- consider fewer octaves or a smaller --octave-scale."
        )

    print(f"\nDreaming {args.image} with params:")
    print(f"model: {args.model}, layer: {layer}, channel: {args.channel}, guide: {args.guide},")
    print(f"octaves: {args.octaves} (scale {args.octave_scale}) -> sizes: {sizes},")
    print(f"iterations: {args.iterations}, step-size: {args.step_size}, jitter: {args.jitter}, max-size: {args.max_size}")
    result = run_octaves(original, sizes, args.iterations, grad_fn, args.step_size, args.jitter)

    save_image(result, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
