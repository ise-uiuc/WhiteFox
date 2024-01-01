
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 10, 5, stride=1, padding=2, dilation=2)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = x2 - 0.6
        x4 = F.relu(x3)
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 50, 50)
