
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 12, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(12)
        self.conv2 = torch.nn.Conv2d(3, 12, 3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(12)
    def forward(self, x):
        v1 = self.conv1(x)
        out1 = self.bn1(v1)
        v2 = self.conv2(x)
        out2 = self.bn2(v2)
        out3 = out1 + out2
        out4 = out3.tanh()
        out5 = torch.nn.functional.relu(out4)
        out6 = out5.add(out3)
        return out6
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
