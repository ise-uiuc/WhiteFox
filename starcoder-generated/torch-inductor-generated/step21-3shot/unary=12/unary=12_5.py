
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.transpose_conv1 = torch.nn.ConvTranspose2d(3, 3, 1, stride=1, padding=0)
        self.transpose_conv2 = torch.nn.ConvTranspose2d(3, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.transpose_conv1(v1, output_size = (1,3,64,64))
        v4 = self.transpose_conv2(v1, output_size = (1,3,64,64))
        return v3 + v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
