import torch
from torch import nn
from .convirt_module import ConVIRTLitModule
from .maskvlm_module import MaskVLMModule


class NLMCXR(torch.nn.Module):
    def __init__(self, convirt_net, convirt_criterion, embedding_dim=512, learning_rate=1e-5, checkpoint_path: str = '', freeze_backbone: bool = True, dropout_rate: float = 0.2):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        # Load pretrained ConVIRT model
        if len(checkpoint_path) != 0:
            self.convirt_model = ConVIRTLitModule.load_from_checkpoint(
                checkpoint_path,
                net=convirt_net,
                criterion=convirt_criterion
            )
        else:
            self.convirt_model = ConVIRTLitModule(
                net=convirt_net,
                criterion=convirt_criterion
            )

        for param in self.convirt_model.parameters():
            param.requires_grad = not self.freeze_backbone

        # Always unfreeze projection layers
        for param in self.convirt_model.net.image_projector.parameters():
            param.requires_grad = True
        for param in self.convirt_model.net.text_projector.parameters():
            param.requires_grad = True            

        # self.adaptation_layer = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.BatchNorm1d(embedding_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(embedding_dim, embedding_dim)
        # )

    def forward(self, input_batch):
        v, u = self.convirt_model(input_batch)

        # # Apply adaptation layer
        # v = self.adaptation_layer(v)
        # u = self.adaptation_layer(u)

        # # Normalize embeddings
        v_normalized = v / v.norm(dim=-1, keepdim=True)
        u_normalized = u / u.norm(dim=-1, keepdim=True)
        
        return v_normalized, u_normalized


class MaskVLMNLMCXR(torch.nn.Module):
    def __init__(self, maskvlm_net, itc_loss, embedding_dim=512, learning_rate=1e-5, checkpoint_path: str = '', freeze_backbone: bool = True, dropout_rate: float = 0.2):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        # Load pretrained ConVIRT model
        self.maskvlm_model = MaskVLMModule.load_from_checkpoint(
            checkpoint_path,
            net=maskvlm_net,
            itc_loss=itc_loss
        )

        for param in self.maskvlm_model.parameters():
            param.requires_grad = not self.freeze_backbone

        # Fine-tune retrieval-related components
        for param in self.maskvlm_model.net.itc_head.parameters():
            param.requires_grad = True

        for param in self.maskvlm_model.net.img_cross_encoder.parameters():
            param.requires_grad = True

        for param in self.maskvlm_model.net.txt_cross_encoder.parameters():
            param.requires_grad = True

        for param in self.maskvlm_model.net.itm_head.parameters():
            param.requires_grad = True 

        # self.adaptation_layer = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.BatchNorm1d(embedding_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(embedding_dim, embedding_dim)
        # )

    def forward(self, input_batch):
        # txt_logits, reconstructed_img, z_img, z_txt, itm_labels, itm_gt = self.maskvlm_model(input_batch)
        
        img_features = self.maskvlm_model.net.image_model(input_batch['image'])
        cls_token = img_features[:, 0, :]

        txt_features = self.maskvlm_model.net.text_model(input_batch['tokenized_data'])
        start_token = txt_features[:, 0, :]

        z_img, z_txt, similarity_matrix = self.maskvlm_model.net.itc_head(cls_token, start_token)

        # # Apply adaptation layer
        # v = self.adaptation_layer(v)
        # u = self.adaptation_layer(u)

        # # Normalize embeddings
        z_img_normalized = z_img / z_img.norm(dim=-1, keepdim=True)
        z_text_normalized = z_txt / z_txt.norm(dim=-1, keepdim=True)
        
        return z_img_normalized, z_text_normalized
