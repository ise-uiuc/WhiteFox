
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pointwise_conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x1, x2):
        v1 = self.pointwise_conv1(x1)
        v2 = self.pointwise_conv1(x1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = self.pointwise_conv1(x1)
        v6 = self.pointwise_conv1(x1)
        v7 = v5 + v6
        v8 = torch.relu(v7)
        v9 = self.pointwise_conv1(x2)
        v10 = self.pointwise_conv1(x2)
        v11 = v9 + v10
        v12 = torch.relu(v11)
        v13 = v4 + v8 + v12
        return v13
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
