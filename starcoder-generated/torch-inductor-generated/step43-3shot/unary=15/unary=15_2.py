
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 6, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 48, 6, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(48, 64, 4, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = torch.relu(v1)
        v5 = torch.relu(v2)
        v6 = torch.relu(v3)
        return v4, v5, v6
# Inputs to the model
x1 = torch.randn(1, 3, 300, 400)
