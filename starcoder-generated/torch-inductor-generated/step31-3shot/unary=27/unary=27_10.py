
class Model(torch.nn.Module):
    def __init__(self, min, max, num):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(num, num, 3, stride=1, padding=1)
        self.pad = torch.nn.ReflectionPad2d((3, 3, 3, 3))
        self.deconv1 = torch.nn.ConvTranspose2d(num, num, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(num, num, 5, stride=1, padding=2)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.pad(v1)
        v3 = self.deconv1(v2)
        v4 = self.conv2(v3)
        v5 = torch.clamp_min(v4, self.min)
        v6 = torch.clamp_max(v5, self.max)
        return v6
num = -9
min = 0.7
max = -0.6
# Inputs to the model
x1 = torch.randn(1, num, 50, 50)
