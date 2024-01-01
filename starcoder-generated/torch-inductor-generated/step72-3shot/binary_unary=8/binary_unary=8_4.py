
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 33, 3, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(33, 53, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 + v1
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)
