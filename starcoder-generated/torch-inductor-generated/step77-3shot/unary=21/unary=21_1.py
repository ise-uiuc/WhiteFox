
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, stride=2, padding=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=1, bias=True, padding_mode='zeros')
        self.conv3 = torch.nn.Conv2d(16, 1, 3, stride=2, padding=1, groups=1, bias=True, padding_mode='zeros')
    def forward(self, x):
        self._padding = self.conv1._padding
        self._stride = self.conv1._stride
        self._dilation = self.conv1._dilation
        v1 = self.conv1(x)
        self._padding = self.conv2._padding
        self._stride = self.conv2._stride
        self._dilation = self.conv2._dilation
        v2 = self.conv2(v1)
        self._padding = self.conv3._padding
        self._stride = self.conv3._stride
        self._dilation = self.conv3._dilation
        v3 = self.conv3(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 1, 10, 10)
