
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 1, stride=5)
        self.conv2 = torch.nn.Conv2d(1, 8, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 - 0.000685
        v4 = F.relu(v3)
        v5 = torch.squeeze(v4, 2)
        v6 = v5.transpose(-1, -2)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
