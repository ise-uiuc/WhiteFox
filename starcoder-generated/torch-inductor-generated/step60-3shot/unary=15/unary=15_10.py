
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 1, 1)
        self.conv2 = torch.nn.Conv2d(2, 1, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(x1)
        v4 = torch.relu(v3)
        return v1, v2, v3, v4
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
