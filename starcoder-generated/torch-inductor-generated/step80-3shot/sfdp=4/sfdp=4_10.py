
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q7, k5, v1, mask):
        qk = q7 @ k5.transpose(-2, -1) / math.sqrt(q7.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v1
        return output
# Inputs to the model
Q0 = torch.randn(1, 64, 56, 56)
K8 = torch.randn(1, 64, 56, 56)
V9 = torch.randn(1, 64, 56, 56)
mask = (torch.randn(1, 56, 56) > 0.7).fill_(-1000000000.0)
