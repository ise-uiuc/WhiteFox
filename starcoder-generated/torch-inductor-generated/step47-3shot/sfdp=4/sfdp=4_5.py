
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, QK, V, mask):
        qK = QK @ V.transpose(-2, -1) / math.sqrt(QK.size(-1))
        qK = qK + mask
        attn_weight = torch.softmax(qK, dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
QK = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
