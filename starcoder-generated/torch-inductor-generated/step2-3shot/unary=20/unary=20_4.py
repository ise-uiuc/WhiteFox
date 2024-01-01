
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.sigmoid = torch.sigmoid
        self.conv_t = torch.nn.ConvTranspose2d(4, 7, kernel_size=5, stride=2)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 16, 16)
