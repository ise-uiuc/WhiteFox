
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0)
        self.conv5 = torch.nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.conv_t = torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.negative_slope = negative_slope
    def forward(self, x2):
        v1 = self.conv1(x2)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        vt1 = self.conv_t(v5)
        vt2 = vt1 > 0
        vt3 = vt1 * self.negative_slope
        vt4 = torch.where(vt2, vt1, vt3)
        return vt4
negative_slope = 0.095887
# Inputs to the model
x2 = torch.randn(2, 3, 113, 106)
