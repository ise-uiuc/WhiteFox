
class Model(nn.Module):
    def __init__(self, min, max):
        super(Model, self).__init__()
        self.min = min
        self.max = max
        self.conv = nn.Conv2d(3, 64, 7, stride=1, padding=2)
    def forward(self, x):
        x = self.conv(x)
        x = torch.clamp_min(x, self.min)
        x = torch.clamp_max(x, self.max)
        return x

def create_ones(*shape):
    return torch.ones(shape, dtype=torch.float32)
min = 1
max = 3
# Inputs to the model
x = create_ones (1, 3, 256, 256)
