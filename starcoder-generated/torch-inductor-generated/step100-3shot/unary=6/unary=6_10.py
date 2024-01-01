
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 11, 2, stride=2, padding=4)
        self.conv2 = torch.nn.Conv2d(11, 7, 2, stride=2, padding=4)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = 3 + v2
        v4 = torch.clamp(v3, 0, 6)
        v5 = v2 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
