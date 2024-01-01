
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 98, 1, stride=1, groups=2)
        self.conv1 = torch.nn.ConvTranspose2d(98, 100, 7, stride=7, bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 161, 161)
