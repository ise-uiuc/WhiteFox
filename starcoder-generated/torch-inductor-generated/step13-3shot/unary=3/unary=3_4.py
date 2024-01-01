
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.BatchNorm2d(1)
    def forward(self, x1):
        v1 = self.norm1(x1)
        v2 = self.norm1(v1)
        v3 = self.norm1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 43, 87)
