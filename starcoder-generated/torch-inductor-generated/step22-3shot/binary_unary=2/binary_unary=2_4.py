
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, 3, stride=1, padding=1)
        self.convx = torch.nn.Conv2d(10, 10, 3, stride=1, padding=1)
        self.convm = torch.nn.Conv2d(1, 10, 3, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.convx(x2)
        v3 = torch.cat([v1, v2], 1)
        v4 = self.convm(v3)
        v5 = v4 - 1.0
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
x2 = torch.randn(1, 1, 28, 28)
