
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(6, 12, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.5
        v3 = v2 - 0.5
        v4 = v3 - 0.5
        v5 = v4 - 0.5
        v6 = v5 - 0.5
        v7 = F.relu(v6)
        v8 = self.conv2(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
