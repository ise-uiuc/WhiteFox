
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 600, 1, stride=1, padding=0)
        self.avgPool = torch.nn.AvgPool2d(kernel_size=4)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.avgPool(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 65, 65)
