
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1, x2):
        v1 = self.sigmoid(x1 + x2)
        v2 = self.sigmoid(x1 - x2)
        v3 = self.sigmoid(x1 * x2)
        v4 = v1 * v2 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
