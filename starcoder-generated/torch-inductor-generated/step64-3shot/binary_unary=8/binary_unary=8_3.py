
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.functional.conv2d
    def forward(self, x1):
        v1 = self.conv(x1, weight=torch.randn(8, 3, 10, 10), bias=None, stride=(2, 3), padding=(1, 2), dilation=(3, 4), groups=2)
        v2 = self.conv(x1, weight=torch.randn(16, 1, 10, 10), bias=None, stride=(4, 5), padding=(0, 0), dilation=(5, 6), groups=1)
        v3 = torch.relu(v1 + v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
