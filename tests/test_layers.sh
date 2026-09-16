#!/bin/bash

# Exit immediately as soon as any one of the below fail
set -e

# usage: bash tests/test_layers.sh $MODEL $CHANNEL $IMAGE (see README.md)
# works with models: mobilenet_v2, mobilenet_v3_large, vgg16, vgg19
MODEL="${1}"
CHANNEL="${2}" # set to -1 to take a mean of all channels instead
IMAGE="${3}"

if [ "$MODEL" != "mobilenet_v2" ] && 
   [ "$MODEL" != "mobilenet_v3_large" ] &&
   [ "$MODEL" != "vgg16" ] &&
   [ "$MODEL" != "vgg19" ]; then
echo "layer_test.sh only knows mobilenet_v2, mobilenet_v3_large, vgg16, vgg19 -- got '$MODEL'" >&2
exit 1
fi

python cli.py --model "$MODEL" --layer "features.1"  --channel "$CHANNEL" --image "$IMAGE" --output dream1.jpg
python cli.py --model "$MODEL" --layer "features.3"  --channel "$CHANNEL" --image "$IMAGE" --output dream3.jpg
python cli.py --model "$MODEL" --layer "features.7"  --channel "$CHANNEL" --image "$IMAGE" --output dream7.jpg
python cli.py --model "$MODEL" --layer "features.13" --channel "$CHANNEL" --image "$IMAGE" --output dream13.jpg
python cli.py --model "$MODEL" --layer "features.16" --channel "$CHANNEL" --image "$IMAGE" --output dream16.jpg
python cli.py --model "$MODEL" --layer "features.18" --channel "$CHANNEL" --image "$IMAGE" --output dream18.jpg
python cli.py --model "$MODEL" --layer "features.21" --channel "$CHANNEL" --image "$IMAGE" --output dream21.jpg
python cli.py --model "$MODEL" --layer "features.24" --channel "$CHANNEL" --image "$IMAGE" --output dream24.jpg
python cli.py --model "$MODEL" --layer "features.27" --channel "$CHANNEL" --image "$IMAGE" --output dream27.jpg
python cli.py --model "$MODEL" --layer "features.30" --channel "$CHANNEL" --image "$IMAGE" --output dream30.jpg
python cli.py --model "$MODEL" --layer "features.33" --channel "$CHANNEL" --image "$IMAGE" --output dream33.jpg
python cli.py --model "$MODEL" --layer "features.36" --channel "$CHANNEL" --image "$IMAGE" --output dream36.jpg
