
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q1, k, V, mask1):
        q2 = torch.randn(1, 64, 56, 56)
        q2 = q + q2
        qk = q1 @ k.transpose(-2, -1) / math.sqrt(q1.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
Q2 = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V1 = torch.randn(1, 64, 56, 56)
mask2 = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
