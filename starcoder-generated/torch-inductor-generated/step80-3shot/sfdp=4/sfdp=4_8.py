
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k4, v, mask, attn):
        qk = q @ k4.transpose(-2, -1) / math.sqrt(q.size(-1)) + attn
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
Q = torch.randn(1, 64, 768)
K = torch.randn(1, 64, 768)
V = torch.randn(1, 64, 768)
mask = (torch.rand(1, 768) > 0.7).fill_(-1000000000.0)
attn = torch.rand(1, 1, 768)  # add attn
