
class Model(torch.nn.Module):
    def __init__(self, weight1=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 3, stride=2, padding=1, _non_persistent_buffers_set=None, weight=weight1, bias=None, dilation=2)
    def forward(self, x):
        v1 = self.conv(x)
        return v1
# Inputs to the model
x = torch.randn(1, 1, 224, 224)
weight = torch.randn(2, 1, 3, 3)
