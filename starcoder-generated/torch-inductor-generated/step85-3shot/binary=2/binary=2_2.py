
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 1, stride=1, padding=0, bias=True)
        self.bn = torch.nn.BatchNorm2d(5)
        self.relu = torch.nn.ReLU6()
        self.pad = torch.nn.ReflectionPad2d(1)
        self.fc = torch.nn.Linear(750, 10)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = self.relu(v2)
        v4 = self.pad(v3)
        v5 = v4.flatten(1)
        v6 = self.fc(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
