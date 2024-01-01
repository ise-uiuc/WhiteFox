
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 6, 2, stride=2, padding=0)
        self.conv_2 = torch.nn.Conv2d(6, 16, 1, stride=1, padding=0)

        self.bn_1 = torch.nn.BatchNorm2d(num_features=6, eps=1e-4, momentum=0.99, affine=True)
        self.bn_2 = torch.nn.BatchNorm2d(num_features=16, eps=1e-4, momentum=0.99, affine=True)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.bn_1(v1)
        v3 = torch.sigmoid(v2)
        v4 = v1 * v3

        v5 = self.conv_2(v4)
        v6 = self.bn_2(v5)
        v7 = torch.sigmoid(v6)
        v8 = v5 * v7

        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
