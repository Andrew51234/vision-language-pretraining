import torch

from .projection import Projector

class ConVIRT(torch.nn.Module):
    def __init__(self,
                 image_model: torch.nn.Module,
                 image_projector: torch.nn.Module,
                 text_model:torch.nn.Module,
                 text_projector: torch.nn.Module,):
        super().__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.image_projector = image_projector
        self.text_projector = text_projector

    def forward(self, input_batch):
        hv = self.image_model(input_batch['image'])
        v = self.image_projector(hv)
        hu = self.text_model(input_batch['tokenized_data'])
        u = self.text_projector(hu)
        return v, u

class ConVIRTForImageClassfication(torch.nn.Module):
    def __init__(self):
        super(ConVIRTForImageClassfication).__init__()
    
    def forward(self):
        pass