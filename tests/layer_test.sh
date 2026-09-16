#!/bin/bash

# Exit immediately as soon as any one of the below fail
set -e

# usage: bash tests/layer_test.sh $MODEL (see README.md)
# works with models: mobilenet_v2, mobilenet_v3_large, vgg16, vgg19
MODEL="${1}"

if [ "$MODEL" != "mobilenet_v2" ] && 
   [ "$MODEL" != "mobilenet_v3_large" ] &&
   [ "$MODEL" != "vgg16" ] &&
   [ "$MODEL" != "vgg19" ]; then
echo "layer_test.sh only knows mobilenet_v2, mobilenet_v3_large, vgg16, vgg19 -- got '$MODEL'" >&2
exit 1
fi

python cli.py --model "$MODEL" --layer "features.1"  --image images/test2.jpg --output dream1.jpg
python cli.py --model "$MODEL" --layer "features.3"  --image images/test2.jpg --output dream3.jpg
python cli.py --model "$MODEL" --layer "features.7"  --image images/test2.jpg --output dream7.jpg
python cli.py --model "$MODEL" --layer "features.13" --image images/test2.jpg --output dream13.jpg
python cli.py --model "$MODEL" --layer "features.16" --image images/test2.jpg --output dream16.jpg
python cli.py --model "$MODEL" --layer "features.18" --image images/test2.jpg --output dream18.jpg
python cli.py --model "$MODEL" --layer "features.21" --image images/test2.jpg --output dream21.jpg
python cli.py --model "$MODEL" --layer "features.24" --image images/test2.jpg --output dream24.jpg
python cli.py --model "$MODEL" --layer "features.27" --image images/test2.jpg --output dream27.jpg
python cli.py --model "$MODEL" --layer "features.30" --image images/test2.jpg --output dream30.jpg
python cli.py --model "$MODEL" --layer "features.33" --image images/test2.jpg --output dream33.jpg
python cli.py --model "$MODEL" --layer "features.36" --image images/test2.jpg --output dream36.jpg
