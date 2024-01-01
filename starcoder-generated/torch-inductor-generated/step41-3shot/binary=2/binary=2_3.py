
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 24, 1, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(24, 32, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = v2 - True
        return v3
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
