
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(40, 64, 3, 1, 0, 1, 1)
        self.conv_t2 = torch.nn.ConvTranspose2d(64, 48, 5, 2, 2, 1, 1)
        self.conv_t3 = torch.nn.ConvTranspose2d(48, 32, 5, 1, 2, 1, 1)
        self.conv_t4 = torch.nn.ConvTranspose2d(32, 4, 1, 1, 0, 1, 1)
    def forward(self, x1):
        v1 = self.conv_t1(x1)
        v2 = self.conv_t2(v1)
        v3 = self.conv_t3(v2)
        v4 = self.conv_t4(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 40, 4, 4)
