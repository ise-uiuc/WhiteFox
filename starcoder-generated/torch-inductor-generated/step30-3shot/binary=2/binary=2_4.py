
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
    def forward(self, x):
        v = self.conv(x)
        return v - 2.373557
# Inputs to the model
x = torch.randn(1, 3, 1, 1)
