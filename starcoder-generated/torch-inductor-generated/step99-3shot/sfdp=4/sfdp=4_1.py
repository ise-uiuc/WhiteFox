
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, i0, m1, k2, q3):
        qk = i0 @ k2.transpose(-2, -1) / math.sqrt(i0.size(-1))
        qk = qk + m1
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ q3
        return output
# Inputs to the model
I = torch.randn(1, 1152, 14, 14)
M = torch.randn(1, 1, 14, 14)
Key = torch.randn(1, 1152, 14, 14)
Query = torch.randn(1, 1152, 14, 14)
