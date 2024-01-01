
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(1, 1, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.relu(v1)
        v3 = torch.cat([v1, v2])
        return v3
# Inputs to the model
x1 = torch.randn(3, 1, 32, 64)
