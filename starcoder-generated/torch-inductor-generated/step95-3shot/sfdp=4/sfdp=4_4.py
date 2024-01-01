
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q,  K, v,  mask):
        QK = q @ K.transpose(-2, -1)
        QK = QK + mask
        attn_weight = torch.softmax(QK, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
