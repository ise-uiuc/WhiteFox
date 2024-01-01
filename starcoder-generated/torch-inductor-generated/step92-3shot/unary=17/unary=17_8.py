
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convl1 = torch.nn.Conv2d(3, 8, (3, 3), 1, 0)
        self.convl2 = torch.nn.Conv2d(8, 16, (3, 3), 1, 1)
        self.convl3 = torch.nn.Conv2d(16, 24, (3, 3), 1, 1)
        self.convl4 = torch.nn.Conv2d(24, 8, (3, 3), 2, 1)
        self.convl5 = torch.nn.ConvTranspose2d(16, 16, (3, 3), 1, 1)
    def forward(self, x2):
        d = "tanh"
        m = "mul"
        v1 = self.convl1(x2)
        v2 = self.convl2(v1)
        v3 = self.convl3(v2)
        v4 = torch.tanh(v3)
        v5 = self.convl4(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.convl5(v6)
        return v7
# Inputs to the model
x2 = torch.randn(1, 3, 32, 32)
