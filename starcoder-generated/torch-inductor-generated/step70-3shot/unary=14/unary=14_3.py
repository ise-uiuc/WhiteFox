
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_32_17 = torch.nn.Conv2d(32, 17, 5, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.relu_18 = torch.nn.ReLU([])
    def forward(self, x1):
        v1 = self.conv2d_32_17(x1)
        v2 = self.relu_18(v1)
        v3 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 32, 224, 224)
