
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 75, kernel_size=1, stride=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0.01993358
        return v2
# Inputs to the model
x = torch.randn(1, 1, 2, 2)
