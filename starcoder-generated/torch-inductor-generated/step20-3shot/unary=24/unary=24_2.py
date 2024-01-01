
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1 > 0
        v3 = v1 * 0.3
        v4 = torch.where(v2, v1, v3)
        v5 = self.conv2(v4)
        v6 = v5 > 0
        v7 = v5 * -0.2
        v8 = torch.where(v6, v5, v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
