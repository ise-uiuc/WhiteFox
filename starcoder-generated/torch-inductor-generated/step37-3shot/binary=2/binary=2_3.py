
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        p1 = 1.4035
        v2 = p1 - v1
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
