
class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1, dilation=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 5, stride=1, padding=2, dilation=2)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = v1 > 0
        v4 = v2 > 0
        v5 = v1 * 0.1
        v7 = v2 * 0.1
        v6 = torch.cat((v1, v2, -v1, -v2), 0)
        v8 = torch.cat((v5, v7), 0)
        v9 = torch.cat((v6, v8), 1)
        v10 = torch.where(v3, v5, v7)
        v11 = torch.where(v4, v9, v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
