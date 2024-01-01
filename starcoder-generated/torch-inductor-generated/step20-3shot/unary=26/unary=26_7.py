
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(480, 471, 2, stride=2)
        self.conv_t2 = torch.nn.ConvTranspose2d(471, 480, 1, stride=1)
    def forward(self, x1):
        x2 = self.conv_t1(x1)
        x3 = self.conv_t2(x2)
        return x3
# Inputs to the model
x1 = torch.randn(32, 480, 16, 16)
