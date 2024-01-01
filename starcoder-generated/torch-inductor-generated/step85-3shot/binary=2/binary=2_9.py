
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 16, kernel_size=(10, 10), stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=(16, 16), stride=1, padding=1)
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = self.conv2(v1)
        v3 = v2 - 1.0
        return v3
# Inputs to the model
x3 = torch.randn(1, 4, 100, 100)
