#!/bin/bash

# Exit immediately as soon as any one of the below fail
set -e

# usage: bash tests/test_grad_params.sh $MODEL $LAYER $CHANNEL $IMAGE (see README.md)
MODEL="${1}"
LAYER="${2}"
CHANNEL="${3}"
IMAGE="${4}"

# Sweeps the gradient-ascent params that aren't model related lik layer/channel:
# --octaves, --octave-scale, --iterations, --step-size, --jitter. Each block below
# varies ONE param around its cli.py default (model/layer/image held fixed)
# so that any visual difference can be attributed to that one change.

# baseline run: all cli.py defaults
python cli.py --model "$MODEL" --layer "$LAYER" --channel "$CHANNEL" --image "$IMAGE" --output dream.params.baseline.jpg


# We've picked low and high values wide apart for visible distinction:
#   - octaves       : how many scales get dreamed -- more octaves layers in
#                      larger, more "fractal" structure on top of the fine
#                      detail the base octave already gives.
python cli.py --model "$MODEL" --layer "$LAYER" --channel "$CHANNEL" --image "$IMAGE" --octaves 1  --output dream.params.octaves.1.jpg
python cli.py --model "$MODEL" --layer "$LAYER" --channel "$CHANNEL" --image "$IMAGE" --octaves 6  --output dream.params.octaves.6.jpg


#   - octave-scale   : size ratio between octaves -- bigger gaps mean each
#                       octave reintroduces coarser structure, changing how
#                       chunky that fractal layering looks.
python cli.py --model "$MODEL" --layer "$LAYER" --channel "$CHANNEL" --image "$IMAGE" --octave-scale 1.2 --output dream.params.octave_scale.1.2.jpg
python cli.py --model "$MODEL" --layer "$LAYER" --channel "$CHANNEL" --image "$IMAGE" --octave-scale 1.8 --output dream.params.octave_scale.1.8.jpg


#   - iterations     : gradient steps per octave -- the main knob for how
#                       strong/saturated the hallucination gets before it
#                       starts overcooking into noise.
python cli.py --model "$MODEL" --layer "$LAYER" --channel "$CHANNEL" --image "$IMAGE" --iterations 5  --output dream.params.iterations.5.jpg
python cli.py --model "$MODEL" --layer "$LAYER" --channel "$CHANNEL" --image "$IMAGE" --iterations 40 --output dream.params.iterations.40.jpg


#   - step-size      : per-step gradient magnitude -- same "how strong" axis
#                       as iterations but coarser-grained, more prone to
#                       blown-out artifacts at the high end.
python cli.py --model "$MODEL" --layer "$LAYER" --channel "$CHANNEL" --image "$IMAGE" --step-size 0.005 --output dream.params.step_size.0.005.jpg
python cli.py --model "$MODEL" --layer "$LAYER" --channel "$CHANNEL" --image "$IMAGE" --step-size 0.03  --output dream.params.step_size.0.03.jpg


#   - jitter         : max random pixel shift before each step -- exists to
#                       stop the output locking onto a repeating tile
#                       pattern; 0 disables it so the artifact it prevents is
#                       visible, default and a larger value show it's fixed.
python cli.py --model "$MODEL" --layer "$LAYER" --channel "$CHANNEL" --image "$IMAGE" --jitter 0  --output dream.params.jitter.0.jpg
python cli.py --model "$MODEL" --layer "$LAYER" --channel "$CHANNEL" --image "$IMAGE" --jitter 32 --output dream.params.jitter.32.jpg
