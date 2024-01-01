
from torch.nn.modules.activation import MultiheadAttention

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiheadAttention()
 
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        v = self.attn(query, key, value, key_padding_mask, attn_mask)[0]
        return v

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 2, 3)
key = torch.randn(2, 4, 8)
value = torch.randn(2, 4, 8)
