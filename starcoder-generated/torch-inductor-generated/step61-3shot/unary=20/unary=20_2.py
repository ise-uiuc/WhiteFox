
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(64)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.bn1(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 128, 256, 256)
