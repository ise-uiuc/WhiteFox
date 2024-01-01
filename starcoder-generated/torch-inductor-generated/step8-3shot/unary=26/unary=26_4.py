
def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class Conv2d_fix_kernel_size(torch.nn.Conv2d):
    __constants__ = [
        "stride", "padding", "dilation", "groups", "padding_mode", "output_padding", "in_channels",
        "out_channels", "kernel_size"
    ]
    kernel_size: Tensor

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: _size_2_t = 0, dilation: _size_2_t = 1, groups: int = 1,
                 bias: bool = True, padding_mode: str = 'zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d_fix_kernel_size, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

    def forward(self, input):
        out = F.conv2d(input, self.weight, self.bias, self.stride, self.padding,
                       self.dilation, self.groups)
        if self.kernel_size == [3, 3] and stride == [2, 2] and self.groups == 1:
            import pdb;pdb.set_trace()
            h, w = out.size()[-2:]
            k = self.kernel_size[-1]
            s = self.stride[-1]
            # 0, 9
            # 1, 10
            # 2, 11
            # 3, 12
            # 4, 13
            # 5, 14
            # 6, 15
            # 7, 16
            # 8, 1
            # 9, 2
            # 10, 3
            # 11, 4
            # 12, 5
            # 13, 6
            # 14, 7
            # 15, 8
            # 8, 17
            # 9, 18
            # 10, 19
            # 11, 20
            # 12, 21
            # 13, 22
            # 14, 23
            # 15, 24
            # 1, 25
            # 2, 26
            # 3, 27
            # 4, 28
            # 5, 29
            # 6, 30
            # 7, 31
            # 8, 25
            # 9, 26
            # 10, 27
            # 11, 28
            # 12, 29
            # 13, 30
            # 14, 31
            # 15, 25
            # 8, 33
            # 9, 34
            # 10, 35
            # 11, 36
            # 12, 37
            # 13, 38
            # 14, 39
            # 15, 40
            # 1, 33
            # 2, 34
            # 3, 35
            # 4, 36
            # 5, 37
            # 6, 38
            # 7, 39
            # 8, 33
            # 9, 34
            # 10, 35
            # 11, 36
            # 12, 37
            # 13, 38
            # 14, 39
            # 15, 40
            return out[:, :, 2:-2, :]
        elif self.kernel_size == [5, 5] and stride == [2, 2] and self.groups == 1:
            h, w = out.size()[-2:]
            k = self.kernel_size[-1]
            s = self.stride[-1]
            # 0, 14
            # 1, 15
            # 2, 16
            # 3, 17
            # 4, 18
            # 5, 19
            # 6, 20
            # 7, 21
            # 8, 22
            # 9, 23
            # 10, 24
            # 11, 14
            # 12, 15
            # 13, 16
            # 14, 17
            # 15, 18
            # 16, 19
            # 17, 20
            # 18, 21
            # 19, 22
            # 20, 23
            # 8, 2
            # 9, 3
            # 10, 4
            # 11, 5
            # 12, 6
            # 13, 7
            # 14, 8
            # 15, 9
            # 16, 10
            # 17, 11
            # 18, 12
            # 19, 13
            # 20, 1
            return out[:, :, 2:-2, :]
        elif self.kernel_size == [11, 11] and stride == [2, 2] and self.groups == 1:

            h, w = out.size()[-2:]
            return out[:, :, 2:-2, :]
        else:
            # TODO: Remove the following two lines once the kernel_size checker has been properly implemented.
            assert type(stride)!= list or bool(set(stride)!= {1}), "The bug exists if stride is a list and set(stride) is not {1}."
            return out


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import pdb;pdb.set_trace()
        self.convs = torch.nn.Sequential(
        Conv2d_fix_kernel_size(1, 1, 3, stride=2, padding=0, groups=1, bias=True),
        Conv2d_fix_kernel_size(1, 1, 5, stride=2, padding=0, groups=1, bias=True),
        Conv2d_fix_kernel_size(1, 1, 11, stride=2, padding=0, groups=1, bias=True))
    def forward(self, x8):
        v1 = self.convs(x8)
        v2 = v1 > 0
        v3 = v1 * -2.2
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x8 = torch.randn([2, 1, 32, 32])
