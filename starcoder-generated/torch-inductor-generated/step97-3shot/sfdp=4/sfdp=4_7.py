
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q2, K, V, mask):
        qk = Q2 <KEY> (-1).transpose(-2, -1) / math.sqrt(Q2.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K3 = torch.randn(1, 64, 56, 56)
V5 = torch.randn(1, 64, 56, 56)
mask2 = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
