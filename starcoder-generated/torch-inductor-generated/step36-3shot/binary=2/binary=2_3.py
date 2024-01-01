
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, kernel_size=2, stride=(1, 1), padding=(0, 0))
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = v1 - [0.0,0.243,0,0,0.913,1711.992,-1,0,0,0,0,0,0,0,0]
        return v2
# Inputs to the model
x3 = torch.randn(1, 3, 4, 4)
