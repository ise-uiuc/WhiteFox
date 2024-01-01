
class Model(torch.nn.Module):
    def __init__(self, axis, min_value=5, max_value=-5, keepdim=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
        self.axis = axis
        self.keepdim = keepdim
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value, {'dim': self.axis, 'keepdim': self.keepdim})
        v3 = torch.clamp_max(v2, self.max_value, {'dim': self.axis, 'keepdim': self.keepdim})
        return v3
axis = 0
min_value = 2
max_value = 1
keepdim = False
# Inputs to the model
x1 = torch.randn(1, 3, 52, 52)
