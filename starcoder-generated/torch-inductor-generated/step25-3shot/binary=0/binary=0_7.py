
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2048, 512, 3, stride=1, padding=1)
    def forward(self, x1,  bias1='default', other='default'):
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 2048, 96, 96)
