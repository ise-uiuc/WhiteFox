
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, d1, v3, qk):
        qk = qk @ d1.transpose(-2, -1) / math.sqrt(qk.size(-1))
        qk = qk + qk
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v3
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
