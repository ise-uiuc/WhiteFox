
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 7, kernel_size=(3, 5), padding=(2, 1), stride=2)
        self.conv_t = torch.nn.ConvTranspose2d(7, 3, kernel_size=3, stride=1)
    def forward(self, x1):
        v7 = self.conv(x1)
        v6 = torch.sigmoid(v7)
        v1 = self.conv_t(v6)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 12, 17)
