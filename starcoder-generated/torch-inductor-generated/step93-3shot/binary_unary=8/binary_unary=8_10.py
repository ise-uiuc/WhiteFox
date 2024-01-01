
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = torch.relu(self.conv1(x1))
        v2 = torch.sigmoid(self.conv1(x1))
        v3 = torch.relu(self.conv1(x1))
        v4 = self.conv1(x1)
        v5 = v2 + v4
        v6 = v3 + v4
        v7 = v1 + v5 + v6
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
