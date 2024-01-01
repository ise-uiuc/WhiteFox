
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 32, kernel_size=1, stride=1, stride=1)
        self.conv_t = torch.nn.ConvTranspose2d(32, 32, kernel_size=4, stride=3, padding=1)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.conv_t(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
