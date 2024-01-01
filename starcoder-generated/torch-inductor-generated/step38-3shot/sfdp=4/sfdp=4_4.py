
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(y, self, query, key, value, attn_mask=0.0):
        return y(self(query, key, value, attn_mask), query, key, value, attn_mask)
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
