
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 24, 9, padding=4)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.maxpool = torch.nn.MaxPool2d(4, stride=4)
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = self.avgpool(v1)
        v1 = self.maxpool(v1)
        v1 = self.gelu(v1)
        t1 = v1.view(v1.shape[0], -1, v1.shape[2], v1.shape[3], v1.shape[4], v1.shape[5]).squeeze()
        return t1
# Inputs to the model
x1 = torch.randn(1, 12, 64, 64)
