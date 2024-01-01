
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 9, kernel_size=7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(9, 24, kernel_size=2, stride=2, padding=0)
        self.conv3 = torch.nn.ConvTranspose2d(24, 12, kernel_size=2, stride=2, padding=0)
    def forward(self, x0):
        v1 = self.conv1(x0)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 > 0
        v5 = v3 * 0.4
        v6 = torch.where(v4, v3, v5)
        return v6
# Inputs to the model
x0 = torch.randn(1, 3, 37, 37)
