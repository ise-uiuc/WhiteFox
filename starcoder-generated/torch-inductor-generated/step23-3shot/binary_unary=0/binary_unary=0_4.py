
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.pool = torch.nn.AvgPool2d(3, 3)
    def forward(self, x5):
        v1 = self.conv(x5)
        v2 = torch.relu(v1)
        v3 = self.pool(v2)
        return v3
# Inputs to the model
x5 = torch.randn(1, 16, 64, 64)
