
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 64, 3, stride=(1, 1), padding=(1, 1))
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 1.67
        return v2
# Inputs to the model
x = torch.randn(8, 1, 34, 34)
