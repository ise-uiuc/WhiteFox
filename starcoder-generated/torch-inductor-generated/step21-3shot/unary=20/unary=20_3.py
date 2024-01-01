
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_t = torch.nn.ConvTranspose2d(1, out_channels=1, kernel_size=[7, 2], stride=[11, 14], padding=[3, 2])
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 189, 465)
