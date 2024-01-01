
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 8, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 1, 17, stride=1, padding=16)
        self.conv3 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1 + x
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 1, 64, 64)
