
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(1, 16, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1 + v1 + v1 + v1
        v4 = v2 + v2 + v2 + v2
        v5 = v3 + v4 + v3 + v4
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
