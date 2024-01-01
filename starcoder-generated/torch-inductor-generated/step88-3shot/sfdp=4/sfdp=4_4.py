
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q5, k9, v, mask):
        qk = q5 @ k9.transpose(-2, -1) / math.sqrt(q5.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
Q5 = torch.randn(1, 64, 56, 56)
K1 = torch.randn(1, 64, 56, 56)
V6 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
