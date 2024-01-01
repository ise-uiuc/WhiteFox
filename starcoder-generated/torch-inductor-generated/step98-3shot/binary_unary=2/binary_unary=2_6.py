
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 45, 11, stride=1, padding=6)
        self.conv2 = torch.nn.Conv2d(45, 2, 26, stride=1, padding=4)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 - 14.7
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(16, 16, 50, 50)
