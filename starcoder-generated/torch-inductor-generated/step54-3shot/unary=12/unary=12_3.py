
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 12, 5, stride=2, padding=1)
        self.conv1 = torch.nn.Conv2d(12, 16, 5, stride=2, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = self.sigmoid(v2)
        v4 = v1 * v3 
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
