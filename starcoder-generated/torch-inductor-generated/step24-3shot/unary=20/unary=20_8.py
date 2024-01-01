
class Model(torch.nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, num_classes, kernel_size=7, stride=2, bias=True, dilation=1)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 49, 64)
