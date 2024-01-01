
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=(3, 3), padding=(4, 3))
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - -0.3033
        return v2
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
