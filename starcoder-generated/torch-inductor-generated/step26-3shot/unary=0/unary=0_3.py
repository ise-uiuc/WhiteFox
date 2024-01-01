
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        padding_list = [0, 0, 1, 1, 1, 1]
        group_list = [1, 12, 1, 1, 7, 7]
        channel_list = [43, 8, 29, 3, 42, 8]
        kernel_size_list = [(7, 3), (1, 3), (3, 21), (1, 1), (3, 1), (3, 2)]
        stride_list = [1, 2, 2, 1, 1, 2]
        for idx, (padding, group, channel, kernel_size, stride) in enumerate(zip(padding_list, group_list, channel_list, kernel_size_list, stride_list)):
            setattr(self, 'conv{}'.format(idx + 1), torch.nn.Conv2d(channel, group, kernel_size, stride=stride, padding=padding, groups=group))
    def forward(self, x3):
        v1 = getattr(self, 'conv1')(x3)
        v11 = self.conv1.weight.view(6, 8, 1, 59, 160).permute(0, 3, 4, 1, 2).contiguous().view(480, 8)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        return v10
# Inputs to the model
x3 = torch.randn(1, 43, 32, 48)
