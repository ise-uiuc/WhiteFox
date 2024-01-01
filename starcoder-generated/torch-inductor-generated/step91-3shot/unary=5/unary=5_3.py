
try:
    x1 = torch.randn(1)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_transpose = torch.nn.ConvTranspose1d(1, 1, 2, stride=2, padding=0)
        def forward(self, _1):
            _10 = self.conv_transpose(_1)
            _20 = _10 * 0.5
            _30 = _10 * 0.7071067811865476
            _40 = torch.erf(_30)
            _50 = _40 + 1
            _60 = _20 * _50
            return _60
    y = Model()(x1)
    print(y)
except TypeError:
    pass

try:
    x1 = torch.randn(1)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_transpose = torch.nn.ConvTranspose1d(1, 1, 4, stride=4, padding=3)
        def forward(self, _1):
            _10 = self.conv_transpose(_1)
            _20 = _10 * 0.5
            _30 = _10 * 0.7071067811865476
            _40 = torch.erf(_30)
            _50 = _40 + 1
            _60 = _20 * _50
            return _60
    y = Model()(x1)
    print(y)
except IndexError:
    pass

try:
    x1 = torch.randn(1)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_transpose = torch.nn.ConvTranspose1d(11, 11, 17, stride=17, padding=0)
        def forward(self, _1):
            _10 = self.conv_transpose(_1)
            _20 = _10 * 0.5
            _30 = _10 * 0.7071067811865476
            _40 = torch.erf(_30)
            _50 = _40 + 1
            _60 = _20 * _50
            return _60
    y = Model()(x1)
    print(y)
except ValueError:
    pass

x1 = torch.randn(1)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(1, 1, 5, stride=5, padding=5)
    def forward(self, _1):
        _10 = self.conv_transpose(_1)
        _20 = _10 * 0.5
        _30 = _10 * 0.7071067811865476
        _40 = torch.erf(_30)
        _50 = _40 + 1
        _60 = _20 * _50
        return _60
y = Model()(x1)
print(y)
