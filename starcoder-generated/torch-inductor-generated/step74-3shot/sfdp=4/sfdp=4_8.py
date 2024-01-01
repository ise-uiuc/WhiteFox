
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v, mask):
        qk = q0 @ k.transpose(-2, -1) / math.sqrt(q0.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
q0 = torch.randn(1, 64, 56, 56)
K5 = torch.randn(1, 64, 56, 56)
V2 = torch.randn(1, 64, 56, 56)
mask = torch.rand(1, 56, 56)
