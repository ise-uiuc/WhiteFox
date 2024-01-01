
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 1, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.nn.ReLU(v1)
        v3 = v2 + x
        v4 = self.conv2(v3)
        v5 = torch.nn.ReLU(v4)
        return v5
# Inputs to the model
x = torch.randn(1, 64, 64, 64)
