#!/bin/bash

# Exit immediately as soon as any one of the below fail
set -e

# usage: bash tests/channel_test.sh $MODEL (see README.md)
# channel counts (checked via a forward-hook probe):
#   - mobilenet_v2 (features.16, 160 channels)
#   - vgg19        (features.30, 512 channels)
MODEL="${1}"

if [ "$MODEL" == "mobilenet_v2" ]; then
python cli.py --model "$MODEL" --layer "features.16" --channel 0   --image images/test2.jpg --output dream.mobilenet_v2.16.0.jpg
python cli.py --model "$MODEL" --layer "features.16" --channel 1   --image images/test2.jpg --output dream.mobilenet_v2.16.1.jpg
python cli.py --model "$MODEL" --layer "features.16" --channel 4   --image images/test2.jpg --output dream.mobilenet_v2.16.4.jpg
python cli.py --model "$MODEL" --layer "features.16" --channel 8   --image images/test2.jpg --output dream.mobilenet_v2.16.8.jpg
python cli.py --model "$MODEL" --layer "features.16" --channel 16  --image images/test2.jpg --output dream.mobilenet_v2.16.16.jpg
python cli.py --model "$MODEL" --layer "features.16" --channel 32  --image images/test2.jpg --output dream.mobilenet_v2.16.32.jpg
python cli.py --model "$MODEL" --layer "features.16" --channel 64  --image images/test2.jpg --output dream.mobilenet_v2.16.64.jpg
python cli.py --model "$MODEL" --layer "features.16" --channel 96  --image images/test2.jpg --output dream.mobilenet_v2.16.96.jpg
python cli.py --model "$MODEL" --layer "features.16" --channel 128 --image images/test2.jpg --output dream.mobilenet_v2.16.128.jpg
python cli.py --model "$MODEL" --layer "features.16" --channel 159 --image images/test2.jpg --output dream.mobilenet_v2.16.159.jpg

elif [ "$MODEL" == "vgg19" ]; then
python cli.py --model "$MODEL" --layer "features.30" --channel 0   --image images/test2.jpg --output dream.vgg19.30.0.jpg
python cli.py --model "$MODEL" --layer "features.30" --channel 1   --image images/test2.jpg --output dream.vgg19.30.1.jpg
python cli.py --model "$MODEL" --layer "features.30" --channel 4   --image images/test2.jpg --output dream.vgg19.30.4.jpg
python cli.py --model "$MODEL" --layer "features.30" --channel 8   --image images/test2.jpg --output dream.vgg19.30.8.jpg
python cli.py --model "$MODEL" --layer "features.30" --channel 16  --image images/test2.jpg --output dream.vgg19.30.16.jpg
python cli.py --model "$MODEL" --layer "features.30" --channel 32  --image images/test2.jpg --output dream.vgg19.30.32.jpg
python cli.py --model "$MODEL" --layer "features.30" --channel 64  --image images/test2.jpg --output dream.vgg19.30.64.jpg
python cli.py --model "$MODEL" --layer "features.30" --channel 96  --image images/test2.jpg --output dream.vgg19.30.96.jpg
python cli.py --model "$MODEL" --layer "features.30" --channel 128 --image images/test2.jpg --output dream.vgg19.30.128.jpg
python cli.py --model "$MODEL" --layer "features.30" --channel 192 --image images/test2.jpg --output dream.vgg19.30.192.jpg
python cli.py --model "$MODEL" --layer "features.30" --channel 256 --image images/test2.jpg --output dream.vgg19.30.256.jpg
python cli.py --model "$MODEL" --layer "features.30" --channel 511 --image images/test2.jpg --output dream.vgg19.30.511.jpg

else
echo "channel_test.sh only knows mobilenet_v2 and vgg19 -- got '$MODEL'" >&2
exit 1
fi
