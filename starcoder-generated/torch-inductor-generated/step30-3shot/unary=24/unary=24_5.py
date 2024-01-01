
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1 > 0
        v3 = v1 * 0.2
        v4 = torch.where(v2, v1, v3)
        v5 = self.conv2(v4)
        v6 = v5 > 0
        v7 = v5 * 0.1
        v8 = torch.where(v6, v5, v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 4, 48, 48)
