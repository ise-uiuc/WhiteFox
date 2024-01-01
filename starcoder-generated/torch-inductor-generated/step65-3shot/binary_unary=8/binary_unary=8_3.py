
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5+v5)
        v7 = torch.relu(v5)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
