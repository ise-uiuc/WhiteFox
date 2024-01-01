
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[4, 4], stride=(2, 2), padding=[2, 2])
    def forward(self, x):
        x = F.sigmoid(F.max_pool2d(x.mul(F.max_pool2d(self.conv(x).mul(F.avg_pool2d(x).mul(F.sigmoid(F.relu(self.conv(x.mul(F.avg_pool2d(x))))))))), kernel_size=3, stride=1, padding=0) + (self.conv(x).mul(F.avg_pool2d(x))).mul((F.avg_pool2d(F.avg_pool2d(self.conv(F.conv(x)).mul(F.avg_pool2d(x)))))))
        return x
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
