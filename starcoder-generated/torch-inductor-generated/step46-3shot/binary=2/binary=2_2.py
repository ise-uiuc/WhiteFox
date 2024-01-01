
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(512, 512, kernel_size=2, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.42
        return v2
# Inputs to the model
x1 = torch.randn(1, 512, 1, 1)
