
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose0 = torch.nn.ConvTranspose2d(1, 64, 9, padding=0, stride=2)
        self.conv2d0 = torch.nn.Conv2d(64, 29, 21, padding=0, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose0(x1)
        v4 = torch.avg_pool2d(v1, 3, stride=1, padding=1)
        v2 = self.conv2d0(v4)
        return v2
# Inputs to the model
x1 = torch.randn(2, 1, 32, 32)
