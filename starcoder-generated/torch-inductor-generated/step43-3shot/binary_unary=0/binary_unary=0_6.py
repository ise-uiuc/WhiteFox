
# This is a wrong model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x, y):
        v1 = self.conv1(x)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = v3 + y
        v5 = torch.relu(v4)
        v6 = self.conv2(v5)
        v7 = v6 + self.conv1(x)
        return v7
# Inputs to the model
