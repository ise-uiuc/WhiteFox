
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q5, K, V, mask):
        qk = Q5 @ K.transpose(-2, -1) / math.sqrt(Q5.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
Q5 = torch.randn(1, 64, 32, 32)
K = torch.randn(1, 64, 32, 32)
V = torch.randn(1, 64, 32, 32)
mask = (torch.rand(1, 32, 32) > 0.7).fill_(-1000000000.0)
