
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 16, (1, 1), stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv2(x1)
        v4 = torch.relu(v1 + v2 + v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
