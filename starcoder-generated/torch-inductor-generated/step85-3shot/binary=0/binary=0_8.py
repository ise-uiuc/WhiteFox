
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 10, 6, stride=1, padding=2, dilation=1)
        self.conv1 = torch.nn.Conv2d(10, 10, 1, stride=1, padding=0, dilation=1)
        self.fc = torch.nn.Linear(80, 10)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = v2.view(100, 80)
        v4 = self.fc(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 10, 64, 64)
x2 = torch.randn(1, 10, 32, 32)
