
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2, stride=1, padding=1)
        self.deconv = torch.nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=2, stride=1, padding=1)
        self.softmax = torch.nn.Softmax(1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + v1
        v3 = self.deconv(v2)
        v4 = v3 + v3
        v5 = self.softmax(v4)
        return v5
# Inputs to the model
x1 = torch.rand(8, 3, 100, 181)
