
from torch import nn

class Model(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in
        self.m = nn.MultiHeadAttention(dim_in, 1)
 
    def forward(self, q1, k2, v3, inv_scale_factor=None, dropout_p=0.):
        q = q1.transpose(-2, -1)
        v = v3.transpose(-2, -1)
        output = self.m(q, k2, v,
                       attn_mask=None,
                       key_padding_mask=None,
                       need_weights=False,
                       static_k=None,
                       static_v=None)[0]
        return output

# Initializing the model
dim_in = 64
m = Model(dim_in)

# Inputs to the model
q1 = torch.randn(1, 64, dim_in)
k2 = torch.randn(1, 16, dim_in)
v3 = torch.randn(1, 16, dim_in)
