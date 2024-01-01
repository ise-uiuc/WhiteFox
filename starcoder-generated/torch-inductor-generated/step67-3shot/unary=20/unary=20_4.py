
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(320, 320, kernel_size=2, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = x1 * v1
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 320, 10, 10, 10)
