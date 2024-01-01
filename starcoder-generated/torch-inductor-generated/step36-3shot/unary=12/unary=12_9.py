
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dilation_rate = 0
        if dilation_rate!= 0:
            p = math.ceil((kernel_size + (dilation_rate - 1) * (kernel_size - 1) - 1) / 2)
        padding2 = max(0, (n + 2 * p - k - (k - 1) * (d - 1))) / 2
        padding_height = padding2
        if padding_height > 0:
            padding_height = math.floor(padding_height)
        else:
            padding_height = math.ceil(padding_height)
        v = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=s, padding=padding,
                            dilation=dilation, groups=1, bias=False)
    def forward(self, x1):
        v1 = self.v(x1)
        v2 = torch.sigmoid(v1)
        return v1 * v2
# Inputs to the model
input_sizes = [
    [1, 3, 16, 16],
    [1, 8, 20, 20],
    [1, 8, 22, 22],
    [1, 16, 50, 50],
    [1, 16, 55, 55],
    [1, 128, 7, 7],
    [1, 128, 9, 9],
    [1, 64, 15, 15],
    [1, 64, 19, 19],
    [1, 128, 29, 29],
    [1, 128, 31, 31],
]
for size in input_sizes:
    x1 = torch.randn(size)
    