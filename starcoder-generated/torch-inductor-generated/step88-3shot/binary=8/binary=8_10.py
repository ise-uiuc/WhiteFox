
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(2)
    def forward(self, x1, x2):
        v1 = self.avgpool(x1)
        v2 = self.avgpool(x2)
        v3 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
x2 = torch.randn(1, 3, 16, 16)
