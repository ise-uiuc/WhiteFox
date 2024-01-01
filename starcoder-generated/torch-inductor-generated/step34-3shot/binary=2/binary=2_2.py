
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 3, stride=-1, padding=0)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 72.65
        z = v2*4.1
        y = z - -4.17
        return y
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
