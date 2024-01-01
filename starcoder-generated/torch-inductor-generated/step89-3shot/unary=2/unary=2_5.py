
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=[384, 3], stride=None, padding=None, dilation=1, return_indices=False, ceil_mode=False)
        self.avg_pool2d = torch.nn.AvgPool2d(kernel_size=[104, 1], stride=None, padding=None, ceil_mode=False, count_include_pad=True)
    def forward(self, x1):
        v1 = self.max_pool2d(x1)
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
x1 = torch.randn(1, 35, 384, 3)
