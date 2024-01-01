
class Model(torch.nn.Module):
    def __init__(self, negative_slope=1.289):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(63, 57, 4, stride=2, padding=3, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x15):
        z1 = self.conv_t(x15)
        z1 = self.relu(z1)
        z1 = self.sigmoid(z1)
        z2 = z1.mean((1, 2))
        z3 = z2 > 0
        z4 = z2 * -9.683
        z5 = torch.where(z3, z2, z4)
        z6 = z5.sum()
        z7 = z5 * 4.959
        return z7
# Inputs to the model
x15 = torch.randn(7, 63, 76, 60)
