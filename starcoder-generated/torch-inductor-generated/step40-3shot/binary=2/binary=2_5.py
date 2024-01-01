
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 9, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 242
        v3 = self.conv2(v2)
        v4 = v3 - 0.353
        return v4
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
