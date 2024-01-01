
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(3, 3, 3)
    def forward(self, x1):
        v1 = self.relu(x1)
        v2 = self.conv(v1)
        v3 = x1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
