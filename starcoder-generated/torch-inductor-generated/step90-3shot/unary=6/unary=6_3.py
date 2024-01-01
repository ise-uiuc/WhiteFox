
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 2, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1).unsqueeze(-1)
        v2 = 3 + v1
        v3 = torch.clamp(v2, 0, 6)
        v4 = self.conv2(x1)
        v5 = v4 * v3
        v6 = v5 / 6
        return v6.unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
