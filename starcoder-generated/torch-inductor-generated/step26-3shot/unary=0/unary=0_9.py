
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128, 4, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
    def forward(self, x2, x3):
        v1 = self.conv1(x3)
        v2 = self.conv2(x2)
        v3 = torch.relu(v2 + v1)
        return v3
# Inputs to the model
x2 = torch.randn(7, 128, 24, 24)
x3 = torch.randn(7, 16, 4, 4)
