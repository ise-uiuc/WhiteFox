
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pointwise_conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1, dilation=1, groups=1, bias=False)
    def forward(self, x1):
        v1 = self.pointwise_conv1(x1)
        v2 = v1 + x1
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
