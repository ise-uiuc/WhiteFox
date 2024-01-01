
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1, stride=1, padding=0, dilation=1, groups=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True, ceil_mode=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2, v3 = self.maxpool(v1)
        v4 = v2 * 0.5
        v5 = v2 * v2
        v6 = v5 * v2
        v7 = v6 * 0.044715
        v8 = v2 + v7
        v9 = v8 * 0.7978845608028654
        v10 = torch.tanh(v9)
        v11 = v10 + 1
        v12 = v4 * v11
        return v12
# Inputs to the model
x1 = torch.randn(1, 3, 196, 196)
