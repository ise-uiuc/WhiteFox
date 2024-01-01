
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(14, 14, 14, 14)
    def forward(self, x):
        v1 = self.avgpool(x)
        v2 = v1 - 1.0
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
