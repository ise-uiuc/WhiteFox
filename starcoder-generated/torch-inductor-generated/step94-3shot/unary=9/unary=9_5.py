
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.relu6 = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.relu6(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
