
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v, attn_mask):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
Q6 = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V6 = torch.randn(1, 64, 56, 56)
mask6 = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
