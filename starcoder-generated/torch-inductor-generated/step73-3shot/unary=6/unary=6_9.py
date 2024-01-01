
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=18)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=20)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 + v1
        v4 = v3.clamp(0, 6)
        v5 = v3 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 230, 230)
