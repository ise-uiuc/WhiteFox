
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(1, 1)
        self.conv1 = torch.nn.Conv2d(3, 3, (3,1))
    def forward(self, x):
        x = self.conv1(x)
        x = self.avgpool(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 4, 6)
