
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=2, dilation=2)
    def forward(self, x):
        v1 = self.conv1(x)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
