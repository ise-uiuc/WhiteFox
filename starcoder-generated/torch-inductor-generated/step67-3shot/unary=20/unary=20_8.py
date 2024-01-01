
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(65, 227, kernel_size=1, stride=1, padding=0)
        self.conv_t1 = torch.nn.ConvTranspose2d(65, 227, dilation=2, kernel_size=1, stride=1, padding=0)
        self.conv_t2 = torch.nn.ConvTranspose2d(65, 227, dilation=3, kernel_size=1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv_t1(x1)
        v4 = torch.sigmoid(v3)
        v5 = self.conv_t2(x1)
        v6 = torch.sigmoid(v5)
        return v2, v4, v6
# Inputs to the model
x1 = torch.randn(1, 65, 226, 226)
