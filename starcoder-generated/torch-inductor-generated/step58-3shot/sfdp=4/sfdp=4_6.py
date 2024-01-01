


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, A1, B3, O3, mask):
        Q = B3 @ A1.transpose(-2, -1) / math.sqrt(A1.size(-1))
        Q = Q + mask
        attn_weight = torch.softmax(Q, dim=-1)
        output = attn_weight @ O3
        return output
      
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
k = torch.randn(1, 64, 56, 56)
v = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
A1 = Q
B3 = k
O3 = v
mask = mask
