
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 4, 3, stride=3, padding=3, dilation=1, groups=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 5, 32, 32)
