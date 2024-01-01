
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(70, 80, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(70, 80, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1 - v2
        m1 = v3 > 0
        v4 = v3 * 0.5
        v5 = torch.where(m1, v3, v4)
        v6 = v3 + v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 70, 64, 64)
