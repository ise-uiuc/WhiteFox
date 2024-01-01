
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, Q, K, V, attn_mask=None):
        QK = torch.matmul(Q, K.permute(0, 1, 3, 2))
        QK = QK / math.sqrt(K.size(-1))
        if attn_mask is not None:
            QK += attn_mask # Add the attention mask
        
        attn_weight = torch.softmax(QK, dim=-1) # Apply softmax to the result
        attn_weight = torch.dropout(attn_weight, 0.1, True) # Apply dropout to the softmax output
        attn_output = torch.matmul(attn_weight, V) # Apply the attention weights to the value
        return attn_output
 
# Initializing the model
m = Model()

# Input tensors of the model
Q = torch.randn(2, 8, 64, 16)
K = torch.randn(2, 8, 16, 64)
V = torch.randn(2, 8, 64, 64)
