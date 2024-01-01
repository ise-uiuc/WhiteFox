
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(40, 17, 1, stride=1, padding=0)
        self.conv_t2 = torch.nn.ConvTranspose2d(17, 40, 3, dilation=3, stride=1, padding=3)
    def forward(self, x1):
        x2 = self.conv_t1(x1)
        x3 = self.conv_t2(x2)
        return x3
# Inputs to the model
x1 = torch.randn(48, 40, 12, 12)
