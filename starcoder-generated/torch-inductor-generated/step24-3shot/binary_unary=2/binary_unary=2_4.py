
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1, groups=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v1 - 0.06
        v4 = F.relu(v2 - torch.tanh(v3))
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
