
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, qk3, V2, mask):
        qk2 = qk3 + mask
        attn_weight = torch.softmax(qk2, dim=-1)
        output = attn_weight @ V2
        return output
# Inputs to the model
K2 = torch.randn(1, 64, 56, 56)
V3 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
