
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q0, kw, v, mask):
        qk = Q0 @ kw.transpose(-2, -1) / math.sqrt(Q0.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
q = torch.randn(1, 64, 192, 192)
kw = torch.randn(1, 64, 192, 192)
v = torch.randn(1, 64, 192, 192)
mask = (torch.rand(1, 192, 192) > 0.7).fill_(-1000000000.0)
