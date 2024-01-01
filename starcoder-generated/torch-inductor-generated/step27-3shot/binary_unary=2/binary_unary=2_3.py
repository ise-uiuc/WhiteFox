
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.add = torch.nn.Add()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 1.25
        v3 = v1.mean([0,2,3])
        v4 = v3 + v2
        v5 = v4 - 0.2
        v6 = self.add(v5, 1.0)
        v7 = F.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
