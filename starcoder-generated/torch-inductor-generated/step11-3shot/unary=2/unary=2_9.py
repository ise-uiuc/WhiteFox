
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = torch.nn.ConvTranspose3d(3, 8, 1, stride=1, padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(8, 14, 1, stride=1, padding=1)
        self.deconv3 = torch.nn.ConvTranspose2d(14, 8, 5, stride=(1, 2), padding=(1, 2))
        self.deconv4 = torch.nn.Conv2d(8, 3, 3, stride=(1, 2), padding=(1, 2))
        self.deconv5 = torch.nn.ConvTranspose1d(16, 16, 32, stride=2, padding=15)
    def forward(self, x1):
        v1 = self.deconv1(x1)
        v2 = self.deconv2(v1)
        v3 = self.deconv3(v2)
        v4 = self.deconv4(v3)
        v5 = self.deconv5(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
