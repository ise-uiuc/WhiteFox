
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 30, 3, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(30, 42, 3, stride=2, padding=2)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = v2 > 0
        v4 = v2 * -0.2
        v5 = torch.where(v3, v2, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 16, 60, 60)
