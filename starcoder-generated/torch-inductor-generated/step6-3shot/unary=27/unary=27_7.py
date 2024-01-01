
class Model(torch.nn.Module):
    def __init__(self, a=None, b=None, c=None, d=None, e=None):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(123, 456, (3, 14))
        self.conv2 = torch.nn.Conv2d(34, 56, (3, 11))
        self.conv3 = torch.nn.Conv2d(354, 678, (1, 1))
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.weight.data.copy_(torch.randn(m.out_channels, m.in_channels, m.kernel_size[0], m.kernel_size[1]))
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

# Inputs
x = torch.randn(size=[1, 123, 1, 14])
