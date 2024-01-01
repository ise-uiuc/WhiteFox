
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v, mask):
        q = torch.einsum(dims, q, k, v)
        qk = q + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
Q = torch.randn(2, 64, 78)
K = torch.randn(2, 96, 32)
V = torch.randn(1, 96, 48)
mask = (torch.rand(1, 2) > 0.7).fill_(-1000000000.0)
