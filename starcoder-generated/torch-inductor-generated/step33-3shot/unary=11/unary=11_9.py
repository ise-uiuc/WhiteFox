
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1 + 10
        v2 = v1 * 20
        v3 = torch.round(v2)
        v4 = v3 - 10
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
