
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 16, 5, stride=1, padding=2)
        self.pool = torch.nn.MaxPool2d(13)
        self.conv2 = torch.nn.Conv2d(16, 18, 5, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.pool(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 6, 104, 104)
