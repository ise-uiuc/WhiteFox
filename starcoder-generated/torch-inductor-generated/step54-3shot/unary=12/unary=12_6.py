
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 5, stride=2, padding=1)
        self.batch_norm = torch.nn.BatchNorm2d(16)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.batch_norm(v1)
        v3 = torch.sigmoid(v2)
        v4 = v2 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
