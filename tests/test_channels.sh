#!/bin/bash

# Exit immediately as soon as any one of the below fail
set -e

# usage: bash tests/test_channels.sh $MODEL (see README.md)
# channel counts:
#   - mobilenet_v2 (features.14, 160 channels)
#   - vgg19        (features.22, 512 channels)
MODEL="${1}"
IMAGE="images/test2.jpg"

if [ "$MODEL" != "mobilenet_v2" ] && [ "$MODEL" != "vgg19" ]; then
echo "test_channels.sh only knows mobilenet_v2 & vgg19 -- got '$MODEL'" >&2
exit 1
fi

if [ "$MODEL" == "mobilenet_v2" ]; then
python cli.py --model "$MODEL" --layer "features.14" --channel 0   --image "$IMAGE" --output dream.mobilenet_v2.14.0.jpg
python cli.py --model "$MODEL" --layer "features.14" --channel 1   --image "$IMAGE" --output dream.mobilenet_v2.14.1.jpg
python cli.py --model "$MODEL" --layer "features.14" --channel 4   --image "$IMAGE" --output dream.mobilenet_v2.14.4.jpg
python cli.py --model "$MODEL" --layer "features.14" --channel 8   --image "$IMAGE" --output dream.mobilenet_v2.14.8.jpg
python cli.py --model "$MODEL" --layer "features.14" --channel 16  --image "$IMAGE" --output dream.mobilenet_v2.14.16.jpg
python cli.py --model "$MODEL" --layer "features.14" --channel 32  --image "$IMAGE" --output dream.mobilenet_v2.14.32.jpg
python cli.py --model "$MODEL" --layer "features.14" --channel 64  --image "$IMAGE" --output dream.mobilenet_v2.14.64.jpg
python cli.py --model "$MODEL" --layer "features.14" --channel 96  --image "$IMAGE" --output dream.mobilenet_v2.14.96.jpg
python cli.py --model "$MODEL" --layer "features.14" --channel 128 --image "$IMAGE" --output dream.mobilenet_v2.14.128.jpg
python cli.py --model "$MODEL" --layer "features.14" --channel 159 --image "$IMAGE" --output dream.mobilenet_v2.14.159.jpg

else
python cli.py --model "$MODEL" --layer "features.22" --channel 0   --image "$IMAGE" --output dream.vgg19.22.0.jpg
python cli.py --model "$MODEL" --layer "features.22" --channel 1   --image "$IMAGE" --output dream.vgg19.22.1.jpg
python cli.py --model "$MODEL" --layer "features.22" --channel 4   --image "$IMAGE" --output dream.vgg19.22.4.jpg
python cli.py --model "$MODEL" --layer "features.22" --channel 8   --image "$IMAGE" --output dream.vgg19.22.8.jpg
python cli.py --model "$MODEL" --layer "features.22" --channel 16  --image "$IMAGE" --output dream.vgg19.22.16.jpg
python cli.py --model "$MODEL" --layer "features.22" --channel 32  --image "$IMAGE" --output dream.vgg19.22.32.jpg
python cli.py --model "$MODEL" --layer "features.22" --channel 64  --image "$IMAGE" --output dream.vgg19.22.64.jpg
python cli.py --model "$MODEL" --layer "features.22" --channel 96  --image "$IMAGE" --output dream.vgg19.22.96.jpg
python cli.py --model "$MODEL" --layer "features.22" --channel 128 --image "$IMAGE" --output dream.vgg19.22.128.jpg
python cli.py --model "$MODEL" --layer "features.22" --channel 192 --image "$IMAGE" --output dream.vgg19.22.192.jpg
python cli.py --model "$MODEL" --layer "features.22" --channel 256 --image "$IMAGE" --output dream.vgg19.22.256.jpg
python cli.py --model "$MODEL" --layer "features.22" --channel 511 --image "$IMAGE" --output dream.vgg19.22.511.jpg
fi
