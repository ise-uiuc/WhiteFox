
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 20, 3, stride=2)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 4.51495885848999
        return v2
# Inputs to the model
x = torch.randn(13, 1, 64, 64)
