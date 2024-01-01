
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1, stride=2)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - x
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
