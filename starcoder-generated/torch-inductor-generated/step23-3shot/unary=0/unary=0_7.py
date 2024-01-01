
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.sigmoid(x1)
        v2 = v1 * 0.08232731481859207
        v3 = x1 * v1
        v4 = v1 * v2
        v5 = v4 + 0.5
        v6 = v5 - 0.4853463692188263
        return v6
# Inputs to the model
x1 = torch.randn(1, 256, 56, 56)
