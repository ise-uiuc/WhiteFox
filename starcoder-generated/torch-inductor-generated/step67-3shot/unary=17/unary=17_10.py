
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding='same')
        self.conv1t = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
    def forward(self, x2):
        b, h, w = x2.shape[0], x2.shape[2], x2.shape[3]
        x2 = self.conv1(x2)
        x2 = self.conv1t(x2)
        return x2
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
