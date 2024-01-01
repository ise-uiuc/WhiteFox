
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(8, 8, 2, stride=2, padding=2)
        self.conv2a = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(8, 8, 2, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv1a(x1)
        v2 = self.conv1b(v1)
        v3 = self.conv2a(v2)
        v4 = self.conv2b(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
