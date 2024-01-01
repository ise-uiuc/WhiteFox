
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(14, 14, kernel_size=4, stride=4, padding=3)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 14, 94, 93)
