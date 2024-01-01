
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        x = self.conv(x)
        return x.clamp(self.min_value, self.max_value)
min_value = 1
max_value = 2
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
