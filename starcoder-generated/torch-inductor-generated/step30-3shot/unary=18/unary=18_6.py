
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool_max = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv = torch.nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.pool_max(x1)
        v2 = self.conv(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 128, 152, 212)
