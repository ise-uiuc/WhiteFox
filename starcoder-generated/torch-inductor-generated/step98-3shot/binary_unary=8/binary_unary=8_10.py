
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pointwise_conv1 = torch.nn.Conv2d(5, 10, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.pointwise_conv2 = torch.nn.Conv2d(5, 10, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.pointwise_conv1(x1)
        v2 = self.pointwise_conv1(x1)
        v3 = self.pointwise_conv1(x1)
        v4 = self.pointwise_conv1(x1)
        v5 = self.pointwise_conv1(x1)
        v6 = v1 + v2 + v3 + v4 + v5
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
