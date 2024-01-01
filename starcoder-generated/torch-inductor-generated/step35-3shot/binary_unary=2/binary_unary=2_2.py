
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(3, 5, 3, stride=3, padding=1))
        self.pool2d = torch.nn.MaxPool2d(2, stride=1, padding=0)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = x2 - 0.5
        x4 = F.relu(x3)
        x5 = self.pool2d(x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
