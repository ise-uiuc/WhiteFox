
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1, x2):
        v1 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)(v3)
        v5 = v4 + x2
        v6 = torch.relu(v5)
        v7 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = 1
