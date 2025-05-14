import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class TextTokenClassifier(nn.Module):
    """Classifier for text tokens based on input features."""

    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, vocab_size)
        init.xavier_uniform_(self.fc.weight)

    def forward(self, wm):
        """
        Forward pass of the classifier.
        
        Args:
            wm (Tensor): Input tensor with shape [B, t_seq, D].
        
        Returns:
            Tensor: Output logits with shape [B, t_seq, vocab_size].
        """
        return self.fc(wm)