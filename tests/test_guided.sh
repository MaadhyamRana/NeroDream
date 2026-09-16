#!/bin/bash

# Exit immediately as soon as any one of the below fail
set -e

# usage: bash tests/test_guided.sh $MODEL $SOURCE_MODEL $GUIDE_MODEL (see README.md)
# works with models: mobilenet_v2, mobilenet_v3_large, vgg16, vgg19
# Exercises --guide (make_guided_grad_fn in deepdream/models.py): source
# image's activation gets pulled toward the guide image's instead of just
# being maximized. Same source/guide pair across a spread of layers --
# early layers should pull mostly texture/color from the guide, deeper
# layers should pull more of its shapes.
MODEL="${1}"
SOURCE_MODEL="${2}"
GUIDE_MODEL="${3}"

if [ "$MODEL" != "mobilenet_v2" ] &&
   [ "$MODEL" != "mobilenet_v3_large" ] &&
   [ "$MODEL" != "vgg16" ] &&
   [ "$MODEL" != "vgg19" ]; then
echo "test_guided.sh only knows mobilenet_v2, mobilenet_v3_large, vgg16, vgg19 -- got '$MODEL'" >&2
exit 1
fi

if [ "$MODEL" == "mobilenet_v2" ]; then
python cli.py --model "$MODEL" --layer "features.3"  --image "$SOURCE_MODEL" --guide "$GUIDE_MODEL" --output dream.guided.mobilenet_v2.3.jpg
python cli.py --model "$MODEL" --layer "features.7"  --image "$SOURCE_MODEL" --guide "$GUIDE_MODEL" --output dream.guided.mobilenet_v2.7.jpg
python cli.py --model "$MODEL" --layer "features.14" --image "$SOURCE_MODEL" --guide "$GUIDE_MODEL" --output dream.guided.mobilenet_v2.14.jpg
python cli.py --model "$MODEL" --layer "features.18" --image "$SOURCE_MODEL" --guide "$GUIDE_MODEL" --output dream.guided.mobilenet_v2.18.jpg

elif [ "$MODEL" == "mobilenet_v3_large" ]; then
python cli.py --model "$MODEL" --layer "features.3"  --image "$SOURCE_MODEL" --guide "$GUIDE_MODEL" --output dream.guided.mobilenet_v3_large.3.jpg
python cli.py --model "$MODEL" --layer "features.7"  --image "$SOURCE_MODEL" --guide "$GUIDE_MODEL" --output dream.guided.mobilenet_v3_large.7.jpg
python cli.py --model "$MODEL" --layer "features.12" --image "$SOURCE_MODEL" --guide "$GUIDE_MODEL" --output dream.guided.mobilenet_v3_large.12.jpg
python cli.py --model "$MODEL" --layer "features.15" --image "$SOURCE_MODEL" --guide "$GUIDE_MODEL" --output dream.guided.mobilenet_v3_large.15.jpg

elif [ "$MODEL" == "vgg16" ]; then
python cli.py --model "$MODEL" --layer "features.7"  --image "$SOURCE_MODEL" --guide "$GUIDE_MODEL" --output dream.guided.vgg16.7.jpg
python cli.py --model "$MODEL" --layer "features.12" --image "$SOURCE_MODEL" --guide "$GUIDE_MODEL" --output dream.guided.vgg16.12.jpg
python cli.py --model "$MODEL" --layer "features.17" --image "$SOURCE_MODEL" --guide "$GUIDE_MODEL" --output dream.guided.vgg16.17.jpg
python cli.py --model "$MODEL" --layer "features.24" --image "$SOURCE_MODEL" --guide "$GUIDE_MODEL" --output dream.guided.vgg16.24.jpg

else
python cli.py --model "$MODEL" --layer "features.7"  --image "$SOURCE_MODEL" --guide "$GUIDE_MODEL" --output dream.guided.vgg19.7.jpg
python cli.py --model "$MODEL" --layer "features.12" --image "$SOURCE_MODEL" --guide "$GUIDE_MODEL" --output dream.guided.vgg19.12.jpg
python cli.py --model "$MODEL" --layer "features.21" --image "$SOURCE_MODEL" --guide "$GUIDE_MODEL" --output dream.guided.vgg19.21.jpg
python cli.py --model "$MODEL" --layer "features.30" --image "$SOURCE_MODEL" --guide "$GUIDE_MODEL" --output dream.guided.vgg19.30.jpg
fi

# Sanity check; --channel should be ignored (with no error) when --guide is given,
# since make_guided_grad_fn's signature has no channel argument.
python cli.py --model "$MODEL" --channel 0 --image "$SOURCE_MODEL" --guide "$GUIDE_MODEL" --output dream.guided.ignores_channel.jpg
