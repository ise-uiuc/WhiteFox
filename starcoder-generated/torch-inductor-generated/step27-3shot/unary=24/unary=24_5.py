
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 3, 4, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 64, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv2(self.conv(x)) > 0
        return v1
# Inputs to the model
x1 = torch.randn(1, 64, 112, 112)
x2 = torch.randn(1, 64, 112, 112)
