
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 80, (224, 224), stride=16, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(v1)
        v3 = self.conv1(v2)
        v4 = self.conv1(v3)
        v5 = self.conv1(v4)
        v6 = v1 + v2 + v3 + v4 + v5
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 3, 224, 224)
