
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q2, k, V, AttentionMask):
        q = Q2
        attn_weight = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.size(-1)), dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
Q3 = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
