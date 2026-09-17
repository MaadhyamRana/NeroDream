#!/bin/bash

# Exit immediately as soon as any one of the below fail
set -e

# usage: bash tests/test_guided.sh $MODEL $SRC_IMG $GUIDE_IMG (see README.md)
# works with models: mobilenet_v2, mobilenet_v3_large, vgg16, vgg19
# Exercises --guide (make_guided_grad_fn in deepdream/models.py): source
# image's activation gets pulled toward the guide image's instead of just
# being maximized. Same source/guide pair across a spread of layers --
# early layers should pull mostly texture/color from the guide, deeper
# layers should pull more of its shapes.
MODEL="${1}"
SRC_IMG="${2}"
GUIDE_IMG="${3}"

if [ "$MODEL" != "" ] &&
   [ "$MODEL" != "mobilenet_v2" ] &&
   [ "$MODEL" != "mobilenet_v3_large" ] &&
   [ "$MODEL" != "vgg16" ] &&
   [ "$MODEL" != "vgg19" ]; then
echo "test_guided.sh only knows default model (inception_v3, leave ""), mobilenet_v2, mobilenet_v3_large, vgg16, vgg19 -- got '$MODEL'" >&2
exit 1
fi

if [ "$MODEL" == "" ]; then
python cli.py  --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.jpg

elif [ "$MODEL" == "mobilenet_v2" ]; then
python cli.py --model "$MODEL" --layer "features.3"  --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.mobilenet_v2.3.jpg
python cli.py --model "$MODEL" --layer "features.7"  --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.mobilenet_v2.7.jpg
python cli.py --model "$MODEL" --layer "features.14" --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.mobilenet_v2.14.jpg
python cli.py --model "$MODEL" --layer "features.18" --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.mobilenet_v2.18.jpg

elif [ "$MODEL" == "mobilenet_v3_large" ]; then
python cli.py --model "$MODEL" --layer "features.3"  --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.mobilenet_v3_large.3.jpg
python cli.py --model "$MODEL" --layer "features.7"  --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.mobilenet_v3_large.7.jpg
python cli.py --model "$MODEL" --layer "features.12" --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.mobilenet_v3_large.12.jpg
python cli.py --model "$MODEL" --layer "features.15" --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.mobilenet_v3_large.15.jpg

elif [ "$MODEL" == "vgg16" ]; then
python cli.py --model "$MODEL" --layer "features.7"  --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.vgg16.7.jpg
python cli.py --model "$MODEL" --layer "features.12" --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.vgg16.12.jpg
python cli.py --model "$MODEL" --layer "features.17" --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.vgg16.17.jpg
python cli.py --model "$MODEL" --layer "features.24" --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.vgg16.24.jpg

else
python cli.py --model "$MODEL" --layer "features.7"  --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.vgg19.7.jpg
python cli.py --model "$MODEL" --layer "features.12" --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.vgg19.12.jpg
python cli.py --model "$MODEL" --layer "features.21" --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.vgg19.21.jpg
python cli.py --model "$MODEL" --layer "features.26" --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.vgg19.21.jpg
python cli.py --model "$MODEL" --layer "features.30" --image "$SRC_IMG" --guide "$GUIDE_IMG" --output images/dream.guided.vgg19.30.jpg
fi
