
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv4 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv4(x1)
        v2 = v1 + x1
        v3 = torch.relu(v2)
        v4 = self.conv5(v3)
        v5 = v4 + v3
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
