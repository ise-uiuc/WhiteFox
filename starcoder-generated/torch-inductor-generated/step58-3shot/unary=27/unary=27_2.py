
class Model(torch.nn.Module):
    def __init__(self, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, kernel_size=(1, 1))
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_max(v1, self.max)
        return v2
max = 20.88
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
