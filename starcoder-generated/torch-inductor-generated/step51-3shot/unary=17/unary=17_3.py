
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(1, 1, 3, stride=1, dilation=2, groups=1, padding=1)
    def forward(self, x1):
        v1 = self.depthwise(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 56, 56)
