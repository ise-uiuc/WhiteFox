
class Model(torch.nn.Module):

    def __init__(self, input_size=128, kernel_size=8,
                 stride=1, padding=3, dilation=1,
                 output_padding=0, groups=1, bias=True):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, groups=groups, bias=bias, padding_mode='zeros')
        self.conv2 = torch.nn.Conv3d(64, input_size, kernel_size, stride=stride, padding=padding,
                             dilation=dilation, groups=groups, bias=bias, padding_mode='zeros')
    def forward(self, x):
        conv1 = self.conv1(x)
        bn = F.batch_norm(conv1, None, None, None, None)
        relu = F.linear(bn, None, None, None, None)
        conv2 = self.conv2(relu)
        return conv2.flatten(1)

# Model begins
class Model(torch.nn.Module):
    def forward(self, inp):
        t1 = torch.mm(inp, inp)
        t2 = torch.mm(inp, inp)
        t3 = torch.mm(inp, t1)
        t4 = t2 + t3
        return t4 + t1

tensor1 = torch.randn(256, 100)
