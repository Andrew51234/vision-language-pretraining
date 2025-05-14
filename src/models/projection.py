import torch
import torch.nn as nn
import torch.nn.functional as F

class Projector(torch.nn.Module):
    def __init__(self, in_dim:int, hidden_size:int, out_dim:int):
        super().__init__()
        self.l1 = torch.nn.Linear(in_dim, hidden_size, bias=False)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, out_dim, bias=False)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.l2(x)

        return x

class ITMHead(nn.Module):
    """
    Image-Text Matching (ITM) Head: Predicts alignment with two logits (aligned or not).

    # TODO model does not learn anything here. Need to debug
    """
    def __init__(self, hidden_size):
        super(ITMHead, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, cls_token, start_token, similarity_matrix):
        """
        Args:
            cls_token: Image `[CLS]` token features (batch_size, hidden_size)
            start_token: Text `[START]` token features (batch_size, hidden_size)
            similarity_matrix: Cosine similarity matrix (batch_size, batch_size)
        Returns:
            labels: Predicted labels based on the fused features (batch_size * 2,)
            gt_labels: Ground truth labels for alignment (batch_size * 2,)
        """
        batch_size = cls_token.size(0)

        # Positive pairs: Element-wise product of aligned features
        positive_features = cls_token * start_token
        positive_labels = torch.ones(batch_size, device=cls_token.device)

        # TODO do we want hard negative mining? Or is random negative fine since we could have the same patient in the batch?
        # Hard negative mining: Identify hardest negatives
        # hard_negative_indices = self.hard_negative_mining(similarity_matrix)
        # select random negatives for now
        hard_negative_indices = torch.randint(0, batch_size, (batch_size, 1), device=cls_token.device)
        # make sure that the negative indices are not the same as the positive indices
        hard_negative_indices = torch.where(hard_negative_indices == torch.arange(batch_size, device=cls_token.device).view(-1, 1), 
                                            (hard_negative_indices + 1) % batch_size, 
                                            hard_negative_indices)
        hard_negative_cls = cls_token[hard_negative_indices.view(-1)]  # Negative image features
        hard_negative_start = start_token[hard_negative_indices.view(-1)]  # Negative text features

        # Negative pairs: Element-wise product of misaligned features
        negative_features = hard_negative_cls * hard_negative_start
        negative_labels = torch.zeros(hard_negative_cls.size(0), device=cls_token.device)

        # Concatenate positive and negative pairs
        fused_features = torch.cat([positive_features, negative_features], dim=0)

        # Concatenate ground truth labels
        gt_labels = torch.cat([positive_labels, negative_labels], dim=0)

        # Predict the alignment based on the fused features
        logits = self.ffn(fused_features).squeeze(-1)
        logits = self.sigmoid(logits)

        return logits, gt_labels

    def hard_negative_mining(self, similarity_matrix, num_negatives=1):
        """
        Perform hard negative mining based on the similarity matrix.
        Args:
            similarity_matrix: Cosine similarity matrix (batch_size, batch_size)
            num_negatives: Number of hard negatives per sample
        Returns:
            negative_indices: Indices of hard negatives (batch_size, num_negatives)
        """
        batch_size = similarity_matrix.size(0)
        negative_indices = []

        for i in range(batch_size):
            similarity_row = similarity_matrix[i].clone()
            similarity_row[i] = -float('inf')  # Exclude the positive pair
            _, hard_negatives = similarity_row.topk(num_negatives, largest=True)
            negative_indices.append(hard_negatives)

        return torch.stack(negative_indices, dim=0)
    
class TripletHead(nn.Module):
    """
    Triplet Head: Predicts alignment with three logits (anchor, positive, negative).
    """
    def __init__(self, img_hidden_size, text_hidden_size, projection_size):
        super(TripletHead, self).__init__()
        self.image_projection = nn.Linear(img_hidden_size, projection_size)
        self.text_projection = nn.Linear(text_hidden_size, projection_size)

    def forward(self, cls_token, start_token):
        """
        Args:
            cls_token: Image `[CLS]` token features (batch_size, hidden_size)
            start_token: Text `[START]` token features (batch_size, hidden_size)
            similarity_matrix: Cosine similarity matrix (batch_size, batch_size)
        Returns:

        """
        img_features = F.normalize(self.image_projection(cls_token), dim=-1)
        txt_features = F.normalize(self.text_projection(start_token), dim=-1)

        anchor = img_features
        positive = txt_features
        # negatives are the shuffled text features
        # TODO: make sure that the negatives are not the same as the positives
        negative_indices = torch.randperm(cls_token.size(0))
        negative = txt_features[negative_indices]

        return anchor, positive, negative

class ITCHead(nn.Module):
    """
    Image-Text Contrastive (ITC) Head: Projects image and text features to a shared space.
    """
    def __init__(self, img_hidden_size, text_hidden_size, projection_size):
        super(ITCHead, self).__init__()
        self.image_projection = nn.Linear(img_hidden_size, projection_size)
        self.text_projection = nn.Linear(text_hidden_size, projection_size)

    def forward(self, cls_token, start_token):
        """
        Args:
            cls_token: Image `[CLS]` token features (batch_size, hidden_size)
            start_token: Text `[START]` token features (batch_size, hidden_size)
        Returns:
            projected_image: Projected image features (batch_size, projection_size)
            projected_text: Projected text features (batch_size, projection_size)
            similarity_matrix: Cosine similarity matrix (batch_size, batch_size)
        """
        # Project features into a shared space
        projected_image = F.normalize(self.image_projection(cls_token), dim=-1)
        projected_text = F.normalize(self.text_projection(start_token), dim=-1)

        similarity_matrix = torch.mm(projected_image, projected_text.t())

        return projected_image, projected_text, similarity_matrix
