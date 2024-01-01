
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = torch.nn.Conv2d(3, 2, 1, stride=1, padding=1)
        self.relu6 = torch.nn.ReLU6()
    def forward(self, x):
        v1 = self.conv1x1(x)
        v2 = self.relu6(v1)
        v3 = v2 - 7
        return v3
# Inputs to the model
x = torch.randn(1, 3, 10, 10)
