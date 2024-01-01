
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(512, 2048, 1, stride=1, padding=0, output_padding=0, groups=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=[20], stride=[20])
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = v1.relu()
        v2 = self.maxpool(v1)
        v3 = v2.sigmoid()
        v4 = v1 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 512, 7, 7)
