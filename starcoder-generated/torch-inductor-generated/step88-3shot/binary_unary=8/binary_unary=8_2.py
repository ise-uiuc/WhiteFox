
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 5, stride=3, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 2, 5, stride=3, padding=0)
        self.conv3 = torch.nn.Conv2d(3, 3, 5, stride=3, padding=0)
    def forward(self, x1):
        v1 = torch.relu(self.conv1(x1))
        v2 = torch.relu(self.conv2(x1))
        v3 = torch.relu(self.conv3(x1))
        v4 = torch.cat([v1, v2, v3], 1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 84, 84)
