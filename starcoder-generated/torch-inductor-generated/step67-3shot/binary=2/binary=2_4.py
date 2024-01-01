
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 1, stride=(0), padding=(0))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 56.5
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
