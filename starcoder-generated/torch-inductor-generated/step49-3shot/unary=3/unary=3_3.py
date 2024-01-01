
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, dilation=2)
    def forward(self, x1):
        v1 = x1
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 129, 129)
