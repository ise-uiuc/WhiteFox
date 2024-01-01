
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(3, 64, 1, stride=2, padding=2)
        self.conv_next = torch.nn.Conv2d(64, 16, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.relu(x1)
        v2 = self.conv(v1)
        v3 = self.relu(v2)
        v2 = self.conv_next(v3)
        v4 = F.sigmoid(v2)
        v4 = v4 * v2
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
