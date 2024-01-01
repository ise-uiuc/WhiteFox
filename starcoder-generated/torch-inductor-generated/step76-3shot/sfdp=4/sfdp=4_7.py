
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V, mask):
        QK = Q @ K.transpose(-2, -1)
        QK = QK / math.sqrt(QK.size(-1))
        QK = QK + mask
        weights = torch.softmax(QK, -1)
        output = weights @ V
        return output
# Inputs to the model
q = torch.randn(1, 64, 56, 56)
k = torch.randn(1, 64, 56, 56)
v = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
