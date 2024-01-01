
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(1, 1)
    def forward(self, x1):
        v1 = self.avgpool(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 40, 40)
