
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 1, 1, stride=1, padding=0)
        self.other = torch.nn.Parameter(torch.randn(1, 1, 1, 1))
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - self.other
        return v2
# Inputs to the model
x = torch.randn(1, 8, 32, 32)
