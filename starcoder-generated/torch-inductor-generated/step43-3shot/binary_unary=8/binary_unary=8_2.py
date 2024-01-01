
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 256, 1, stride=2, bias=False)
        self.conv2 = torch.nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False)
    def forward(self, x1):
        v1 = torch.relu(self.conv1(x1))
        v2 = self.conv2(v1)
        v3 = torch.relu(v2)
        v4 = torch.relu(self.conv2(v1))
        v5 = v4 + v3
        v6 = torch.relu(v5)
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
