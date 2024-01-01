
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, qK, v, mask):
        q = torch.sigmoid(qk @ v.transpose(-2, -1) / math.sqrt(qk.size(-1)) + mask)
        attn_weight = torch.softmax(q, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
qK2 = torch.randn(1, 64, 56, 56)
v = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
