
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        _X = self.conv(x1)
        _0 = torch.nn.functional.relu6(_X + 3) / 6
        _1 = _0 / 3
        _2 = _1 * 3
        return _2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
