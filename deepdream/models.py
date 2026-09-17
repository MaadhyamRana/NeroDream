"""Pretrained registry for models to use.

Every entry is a torchvision classifier pretrained on ImageNet. Weights
download automatically the first time you use a given model (needs internet
once) and are cached to ~/.cache/torch/hub/checkpoints/ after that.

We only need to expose the gradient function to the rest of the project, so models
are all handled here first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
import torchvision.models as tvm
from torchvision.models import (
    EfficientNet_B0_Weights,
    GoogLeNet_Weights,
    Inception_V3_Weights,
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    VGG16_Weights,
    VGG19_Weights,
)


@dataclass
class ModelEntry:
    loader: Callable[[], torch.nn.Module]
    default_layer: str
    approx_size_mb: int
    min_recommended_input_px: int
    note: str


# Smallest-octave sizes below min_recommended_input_px risk the network's
# internal pooling shrinking a feature map to zero -- keep your smallest
# octave at or above that value for a given model.
AVAILABLE_MODELS = {
    "mobilenet_v2": ModelEntry(
        lambda: tvm.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT),
        "features.14",
        14,
        32,
        "Tiny, fast, forgiving of small octave sizes. Good default.",
    ),
    "mobilenet_v3_large": ModelEntry(
        lambda: tvm.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT),
        "features.12",
        22,
        32,
        "Similar size to v2, slightly different texture bias.",
    ),
    "googlenet": ModelEntry(
        lambda: tvm.googlenet(weights=GoogLeNet_Weights.DEFAULT),
        "inception4c",
        50,
        64,
        "Same lineage as the original DeepDream network, smaller than inception_v3.",
    ),
    "resnet50": ModelEntry(
        lambda: tvm.resnet50(weights=ResNet50_Weights.DEFAULT),
        "layer3",
        98,
        64,
        "Residual connections give a more abstract, less 'eyeball-y' hallucination style.",
    ),
    "inception_v3": ModelEntry(
        lambda: tvm.inception_v3(weights=Inception_V3_Weights.DEFAULT),
        "Mixed_6c.branch1x1",
        104,
        75,
        "The original DeepDream backbone. Classic swirly eyes/dog-face look.",
    ),
    "efficientnet_b0": ModelEntry(
        lambda: tvm.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT),
        "features.6",
        20,
        64,
        "Good quality-per-MB, less commonly used for DeepDream and more experimental.",
    ),
    "vgg16": ModelEntry(
        lambda: tvm.vgg16(weights=VGG16_Weights.DEFAULT),
        "features.17",
        528,
        32,
        "Clean sequential conv stack; standard neural-style-transfer's backbone",
    ),
    "vgg19": ModelEntry(
        lambda: tvm.vgg19(weights=VGG19_Weights.DEFAULT),
        "features.21",
        548,
        32,
        "Deeper VGG variant, slightly richer textures than vgg16. Large download.",
    ),
}


def get_model(name: str) -> torch.nn.Module:
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model '{name}'. Options: {list(AVAILABLE_MODELS)}")
    model = AVAILABLE_MODELS[name].loader()
    model.eval()
    
    # freeze the model weights as we aren't training
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def list_layers(model: torch.nn.Module) -> List[str]:
    """Every named module in the model -- use this to find alternative layer
    names to try for a given architecture."""
    return [name for name, _ in model.named_modules() if name]


def make_torch_grad_fn(
    model: torch.nn.Module, layer_name: str, channel: Optional[int] = None
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a compute_grad_fn(image_hwc) -> gradient_hwc closure for core.run_octaves.

    Hooks `layer_name`, maximizes the mean square of its activation (or a
    single channel's, if `channel` is given), and returns d(loss)/d(pixels)
    as a plain HWC numpy array, exactly the interface core.py expects.
    """
    named_modules = dict(model.named_modules())
    if layer_name not in named_modules:
        raise ValueError(f"No layer '{layer_name}' in this model. Try list_layers(model).")
    target_module = named_modules[layer_name]

    """
    constants for the ImageNet dataset, for normalization below.
    matches the shape of tensor required by torch CNN models.
    
    DISABLE/REPLACE if not using model trained on ImageNet's distribution.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


    """
    This is so that every time the line below "model(normalized)" runs,
    we can fill up the activation dictionary above with the outputs of 
    what the layer 'responds' with...
    """
    activation = {}
    def hook(_module, _input, output):
        activation["value"] = output
    target_module.register_forward_hook(hook)


    def compute_grad_fn(image_hwc: np.ndarray) -> np.ndarray:
        """
        Every torch CNN expects channel-first (CHW), not channel-last (HWC),
        and a batch dimension so we transpose the tensor (image) axes first.
        
        The core DeepDream effect:
        Turning the gradient flag on for the input tensor, not on a weight of model
        (which is itself frozen). So when we call loss.backward() later, PyTorch computes 
        the gradient of how would nudging each pixel changes the loss, as opposed to nudging
        weight in the model.
        """
        tensor = torch.from_numpy(image_hwc).permute(2, 0, 1).unsqueeze(0).float()
        tensor.requires_grad_(True)
        
        """
        forward pass after normalization for the layer.
        Use model(tensor) in case not using ImageNet based model / apply the suitable normalization.
        
        We dont care for the output of the model itself, just its internal activations that
        helps us exaggerate the features at this layer back in gradient ascent.
        """
        normalized = (tensor - mean) / std
        model(normalized)
        act = activation["value"]
        
        """
        Now run the backward pass to use the retrieved responses of the layer and ... dream.
         
        Channel picking is done to pick a filter's personality instead of the average of the layer.
        Squaring means both +ve and -ve activations count as +ve activity;
        gradient ascent pushes magnitude up regardless of direction.
        """
        target = act[:, channel] if (channel != -1) else act
        loss = target.pow(2).mean()
        loss.backward()
        return tensor.grad.squeeze(0).permute(1, 2, 0).numpy() # restore axes again to HWC

    return compute_grad_fn


def make_guided_grad_fn(
    model: torch.nn.Module, layer_name: str, guide_image_hwc: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Same compute_grad_fn(image_hwc) -> gradient_hwc interface as
    make_torch_grad_fn, but instead of maximizing the layer's activation in
    general, pulls the source's activation at that layer *toward* the
    guide image's -- so the dream hallucinates shapes pulled from the guide
    instead of generic ImageNet classes.

    The guide's activation is averaged over space into one target vector
    per channel, so it stays comparable to the source's activation no
    matter how the source's resolution (and so its activation's spatial
    size) changes across octaves.
    """
    named_modules = dict(model.named_modules())
    if layer_name not in named_modules:
        raise ValueError(f"No layer '{layer_name}' in this model. Try list_layers(model).")
    target_module = named_modules[layer_name]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    activation = {}
    def hook(_module, _input, output):
        activation["value"] = output
    target_module.register_forward_hook(hook)

    # One-off forward pass on the guide image to get its (fixed) target activation.
    guide_tensor = torch.from_numpy(guide_image_hwc).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
        model((guide_tensor - mean) / std)
    guide_target = activation["value"].mean(dim=(2, 3), keepdim=True)

    def compute_grad_fn(image_hwc: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(image_hwc).permute(2, 0, 1).unsqueeze(0).float()
        tensor.requires_grad_(True)

        normalized = (tensor - mean) / std
        model(normalized)
        act = activation["value"]

        """
        core.py's gradient_ascent_step always *adds* step_size * grad, i.e.
        it only knows how to ascend. To pull the source's activation toward
        the guide's (minimize distance) through that same ascent-only
        interface, we ascend on the negated distance instead -- negating
        the loss before backward() flips the returned gradient's sign, so
        adding it is equivalent to descending the real (unnegated) distance.
        """
        loss = -((act - guide_target) ** 2).mean()
        loss.backward()
        return tensor.grad.squeeze(0).permute(1, 2, 0).numpy()

    return compute_grad_fn
