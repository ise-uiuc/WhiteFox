
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1, dilation=[1, 2, 3, 4])
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, dilation=1, groups=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        if v1.shape!= v2.shape:
            v2 = F.interpolate(v2, size=(v1.shape[2], v1.shape[3]), mode="nearest")
            v2 = v2.expand(v1.shape)
        v3 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(2, 3, 128, 256)
