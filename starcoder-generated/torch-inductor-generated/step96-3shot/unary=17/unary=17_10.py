
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(16, 32, 3, 2, 1, bias=True)
        self.conv_1 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=True)
        self.conv_2 = torch.nn.Conv2d(32, 16, 3, stride=2, padding=1, bias=True)
        self.conv_3 = torch.nn.Conv2d(16, 3, 3, stride=2, padding=1, bias=True)
    def forward(self, x):
        v = self.conv_t(x)
        v1 = self.conv_1(v)
        v2 = self.conv_2(v1)
        v3 = self.conv_3(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 16, 32, 32)
