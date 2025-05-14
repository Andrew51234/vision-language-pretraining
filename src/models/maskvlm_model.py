import torch

from .text_encoder import MVLMTextEncoder
from .image_encoder import MaskVLMImageEncoder
from .components.cross_modality import TxtCrossModalityEncoder, ImgCrossModalityEncoder, ImgCrossModalityDecoder
from .components.txt_decoder import TextTokenClassifier
from .projection import Projector, ITCHead, ITMHead

class MaskVLM(torch.nn.Module):
    def __init__(self,
                 image_model: MaskVLMImageEncoder,
                 text_model: MVLMTextEncoder,
                 itc_head: ITCHead,
                 itm_head: ITMHead,
                 img_cross_encoder: ImgCrossModalityEncoder,
                 txt_cross_encoder: TxtCrossModalityEncoder,
                 img_cross_decoder: ImgCrossModalityDecoder,
                 txt_classifier: TextTokenClassifier):
        super().__init__()
        # Initialize the model components
        ### image related
        self.image_model = image_model
        self.img_cross_encoder = img_cross_encoder
        self.img_cross_decoder = img_cross_decoder
        
        ### text related
        self.text_model = text_model
        self.txt_cross_encoder = txt_cross_encoder
        self.txt_classifier = txt_classifier

        # ITC loss components
        self.itc_head = itc_head
        self.itm_head = itm_head

    def forward(self, input_batch):

        # Extract features from the image and text
        # [B, 3, H, W] → [B, v_seq, D]
        img_features = self.image_model(input_batch['image'])
        masked_img_features = self.image_model(input_batch['masked_img'])
        cls_token = img_features[:, 0, :]
        # print(cls_token.shape)

        # → [B, t_seq, D]
        txt_features = self.text_model(input_batch['tokenized_data'])
        masked_txt_features = self.text_model(input_batch['masked_tokenized_data'])
        start_token = txt_features[:, 0, :]
        # print(start_token.shape)

        # Project image and text features to a shared space for ITC loss
        z_img, z_txt, similarity_matrix = self.itc_head(cls_token, start_token)

        # Cross-attention
        vm = self.img_cross_encoder(
            masked_img_features,
            txt_features,
            input_batch['text_mask']
        ) # [B, v_seq, D]
        z_img_cross = vm[:, 0, :]

        wm = self.txt_cross_encoder(
            masked_txt_features,
            img_features,
            input_batch['text_mask']
        ) # [B, t_seq, D]
        z_txt_cross = wm[:, 0, :]

        # ITM loss preparation
        itm_labels, itm_gt = self.itm_head(z_img_cross, z_txt_cross, similarity_matrix)

        # Text classification
        # [B, t_seq, D] → [B, t_seq, vocab_size]
        txt_logits = self.txt_classifier(wm)

        # Image reconstruction grayscale not rgb
        # [B, v_seq, D] → [B, v_seq, patch_size * patch_size * 1]
        reconstructed_img = self.img_cross_decoder(
            vm,
            txt_features,
            input_batch['text_mask']
        )

        return txt_logits, reconstructed_img, z_img, z_txt, itm_labels, itm_gt
