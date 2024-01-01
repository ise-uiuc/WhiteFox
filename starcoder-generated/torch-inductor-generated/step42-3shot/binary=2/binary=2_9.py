
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=False)
    def forward(self, x):
        v1 = self.avgpool(x)
        v2 = v1 - torch.nn.Parameter(0.3070)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
