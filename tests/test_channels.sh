#!/bin/bash

# Exit immediately as soon as any one of the below fail
set -e

# usage: bash tests/test_channels.sh $MODEL $LAYER $IMAGE (see README.md and deepdream/models.py)
MODEL="${1}"
LAYER="${2}"
IMAGE="${3}"

python cli.py --model "$MODEL" --layer "$LAYER" --channel 0   --image "$IMAGE" --output dream.CH0.jpg
python cli.py --model "$MODEL" --layer "$LAYER" --channel 1   --image "$IMAGE" --output dream.CH1.jpg
python cli.py --model "$MODEL" --layer "$LAYER" --channel 4   --image "$IMAGE" --output dream.CH4.jpg
python cli.py --model "$MODEL" --layer "$LAYER" --channel 8   --image "$IMAGE" --output dream.CH8.jpg
python cli.py --model "$MODEL" --layer "$LAYER" --channel 16  --image "$IMAGE" --output dream.CH16.jpg
python cli.py --model "$MODEL" --layer "$LAYER" --channel 32  --image "$IMAGE" --output dream.CH32.jpg
python cli.py --model "$MODEL" --layer "$LAYER" --channel 64  --image "$IMAGE" --output dream.CH64.jpg
python cli.py --model "$MODEL" --layer "$LAYER" --channel 96  --image "$IMAGE" --output dream.CH96.jpg
python cli.py --model "$MODEL" --layer "$LAYER" --channel 128 --image "$IMAGE" --output dream.CH128.jpg
python cli.py --model "$MODEL" --layer "$LAYER" --channel 159 --image "$IMAGE" --output dream.CH159.jpg
python cli.py --model "$MODEL" --layer "$LAYER" --channel 192 --image "$IMAGE" --output dream.CH192.jpg
python cli.py --model "$MODEL" --layer "$LAYER" --channel 256 --image "$IMAGE" --output dream.CH256.jpg
python cli.py --model "$MODEL" --layer "$LAYER" --channel 511 --image "$IMAGE" --output dream.CH511.jpg

