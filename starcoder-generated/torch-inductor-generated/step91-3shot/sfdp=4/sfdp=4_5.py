
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, qK, v3, mask):
        qK_ = qK @ v3.transpose(-2, -1) / math.sqrt(qK.size(-1))
        qK_ = qK_ + mask
        attn_weight = torch.softmax(qK_, dim=-1)
        output = attn_weight @ v3
        return output
# Inputs to the model
qk1 = torch.randn(1, 64, 56, 56)
v3 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
