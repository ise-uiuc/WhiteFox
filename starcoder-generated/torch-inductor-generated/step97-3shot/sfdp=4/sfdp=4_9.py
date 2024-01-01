
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q1, k2, V2, mask2):
        qk2 = Q1 @ k2.transpose(-2, -1) / math.sqrt(Q1.size(-1))
        qk2 = qk2 + mask2
        attn_weight1 = torch.softmax(qk2, dim=-1)
        output = attn_weight1 @ V2
        return output
# Inputs to the model
Q9 = torch.randn(1, 64, 56, 56)
k2 = torch.randn(1, 64, 56, 56)
V2 = torch.randn(1, 64, 56, 56)
mask2 = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
