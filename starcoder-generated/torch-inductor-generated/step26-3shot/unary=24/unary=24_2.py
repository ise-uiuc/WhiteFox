
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv = torch.nn.Conv2d(256, 1, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU6(inplace=False)
        self.batch_norm = torch.nn.BatchNorm2d(1, eps=0.001, momentum=0.03, affine=True)
        self.negative_slope = negative_slope
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.batch_norm(v1)
        v3 = self.relu(v2)
        v4 = v1 > 0
        v5 = v1 * self.negative_slope
        v6 = torch.where(v4, v1, v5)
        return v6
negative_slope = 0.1
# Inputs to the model
x1 = torch.randn(1, 256, 14, 14)
