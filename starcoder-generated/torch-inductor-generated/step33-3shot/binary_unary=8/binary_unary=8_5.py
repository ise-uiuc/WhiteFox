
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)
        self.conv3 = torch.nn.Conv2d(1, 1, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = torch.relu(v1 + v2 + v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
