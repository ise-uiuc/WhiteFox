
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(3)
        self.conv = torch.nn.Conv2d(8, 16, 5, stride=3, padding=1)
    def forward(self, x1):
        v0 = self.pool(x1)
        v1 = self.conv(v0)
        v2 = v1 - 1
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 160, 160)
