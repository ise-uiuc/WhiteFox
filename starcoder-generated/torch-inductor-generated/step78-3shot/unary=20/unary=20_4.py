
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.tanh(v1)
        v3 = torch.sigmoid(v2)
        v4 = torch.sigmoid(v2)
        return v3, v4
# Inputs to the model
x1 = torch.randn(1, 32, 256, 256)
