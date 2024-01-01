
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 3, kernel_size=3, stride=2)
    def forward(self, x1):
        v1 = torch.sigmoid(self.conv_t(x1))
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 11, 11)
