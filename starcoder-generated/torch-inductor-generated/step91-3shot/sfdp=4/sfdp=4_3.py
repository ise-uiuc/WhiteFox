
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, W09, M8, v1, mask):
        qk = W09 @ M8.transpose(-2, -1) / math.sqrt(W09.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v1
        return output
# Inputs to the model
W6 = torch.randn(1, 64, 56, 56)
M7 = torch.randn(1, 56, 56)
V2 = torch.randn(1, 56, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
