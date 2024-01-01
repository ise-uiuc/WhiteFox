
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1.max(1, True)[0]
        v3 = v2 > 0
        v4 = v2 * -0.1
        v5 = torch.where(v3, v2, v4)
        return self.conv2(v5)
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
