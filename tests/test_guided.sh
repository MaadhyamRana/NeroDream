#!/bin/bash

# Exit immediately as soon as any one of the below fail
set -e

# usage: bash tests/test_guided.sh $SRC_IMG $GUIDE_IMG
# (Smaller models work as this is a heavier task)
SRC_IMG="${1}"
GUIDE_IMG="${2}"

# test on different models
python cli.py --iterations 50 --model "mobilenet_v2" --layer "features.7"  --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.mobilenet_v2.7.jpg
python cli.py --iterations 50 --model "mobilenet_v3_large" --layer "features.7"  --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.mobilenet_v3_large.7.jpg
python cli.py --iterations 50 --model "efficientnet_b0" --layer "features.6"  --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.efficientnet_b0.6.jpg
