
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 1, padding=1)
        self.conv2 = torch.nn.Conv2d(6, 2, 1, padding=2, dilation=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 - 50
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 40, 60)
