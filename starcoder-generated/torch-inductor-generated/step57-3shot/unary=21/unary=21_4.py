
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 16, 5, padding=0, bias=False, groups=2)
        self.conv2 = torch.nn.Conv2d(16, 256, 1)
        self.conv3 = torch.nn.Conv2d(32, 2, kernel_size=1)
    def forward(self, x):
        _t1 = self.conv1(x)
        _t = torch.tanh(_t1)
        _t = self.conv2(_t) + torch.sin(_t1) + torch.cos(_t)
        _t = (_t - _t.abs()) * torch.sigmoid(_t)
        _t = _t * _t
        _t = torch.relu(_t) * torch.sigmoid(_t)
        _t = torch.tanh(_t) + torch.sigmoid(_t) - torch.tanh(_t) + torch.sin(x).exp()
        _t2 = self.conv3(x) + torch.tan(_t1.tanh()) + torch.sigmoid(_t)
        return _t2
# Inputs to the model
x = torch.randn(1, 32, 24, 24)
