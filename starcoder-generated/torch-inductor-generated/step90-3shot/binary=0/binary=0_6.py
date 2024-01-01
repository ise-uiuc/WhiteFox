
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 12, 1, stride=1, padding=1)
    def forward(self, x, kernel_size=3, stride=1, padding=0):
        v1 = self.conv(x)
        return v1
# Inputs to the model
x = torch.randn(1, 1, 28, 28)
