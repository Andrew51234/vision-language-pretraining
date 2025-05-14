from typing import Optional, Any
import torch
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from transformers import ViTModel

class ImageEncoder(torch.nn.Module):
    def __init__(self,
                 name: str,
                 frozen: bool=False,
                 weights: Optional[Any]=None,
                 return_interm_layers: bool=False):
        super().__init__()
        backbone = getattr(torchvision.models, name)(weights=weights)

        ## Setting layers to be non-trainable
        # for name, parameter in backbone.named_parameters():
        #     if frozen or "layer2" not in name and "layer3" not in name and "layer4" not in name:
        #         parameter.requires_grad_(False)

        if return_interm_layers:
            return_layers = {"layer1": "0",
                             "layer2": "1",
                             "layer3": "2",
                             "layer4": "3", 
                             "avgpool": "hv"}
        else:
            return_layers = {"layer4": "0",
                             "avgpool": "hv"}

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def reload_model_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path))

    def forward(self, images):
        out = self.body(images)
        ## Get the output of the avgpool (B, 2048, 1, 1)
        return out["hv"].squeeze()

class VitImageEncoder(torch.nn.Module):
    """
    VitImageEncoder is a wrapper around the Vision Transformer (ViT) model from the Hugging Face library.
    It uses the CLS token from the last hidden state as the image representation.
    Attributes:
        model (ViTModel): The Vision Transformer model.
    Methods:
        __init__(name: str):
            Initializes the VitImageEncoder with a pretrained ViT model.
        forward(images):
            Forward pass for the image encoder. Takes a batch of images and returns the CLS token representation.
        reload_model_weights(weights_path):
            Reloads the model weights from a specified file path.
    """
    def __init__(self, name: str):
        super().__init__()
        self.model = ViTModel.from_pretrained(name)
        # self.processor = ViTImageProcessor.from_pretrained(name)

    def forward(self, images):
        tokens = self.model(images) # inputs['pixel_values'])
        # Use the first token (CLS token) as the image representation
        hv = tokens.last_hidden_state[:, 0, :] # (1, 768) in case of ViT
        return hv
    
    def reload_model_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path))

class MaskVLMImageEncoder(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.model = ViTModel.from_pretrained(name)
        # for maskvlm, we don't freeze any layers

    def forward(self, images):
        tokens = self.model(images) # inputs['pixel_values'])
        return tokens.last_hidden_state
    
    def reload_model_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path))
