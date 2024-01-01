
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(2, 15, 2, stride=2)
        self.conv_t2 = torch.nn.ConvTranspose2d(15, 1024, 3, stride=3)
        self.conv_t3 = torch.nn.ConvTranspose2d(1024, 6, 2, stride=2)
    def forward(self, x1):
        x2 = self.conv_t1(x1)
        x3 = x2 + 0.06506
        x4 = x3 * 2.9049
        x5 = torch.sigmoid(x4)
        x6 = x5 * x3
        x7 = self.conv_t2(x6)
        x8 = torch.pow(x3, 2.7487)
        x9 = self.conv_t3(x8)
        x10 = x9 * x7
        x11 = x10 - 0.25819
        return x11
# Inputs to the model
x1 = torch.randn(7, 2, 8, 8)
