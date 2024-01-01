
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = v2 * v1
        v4 = self.relu(x1)
        v5 = v3 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
