
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(2, 1, 5, dtype=float)
        self.conv2d_0 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(1)
    def forward(self, x2):
        x = x2
        x4 = self.conv2d(x)
        x5 = self.conv2d_0(x4)
        x7 = x5.contiguous()
        x6 = x7.transpose(2, 3)
        v1 = self.adaptive_avg_pool2d(x5)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        return v9
# Inputs to the model
x2 = torch.randn(1, 2, 28, 28)
