
class Model(torch.nn.Module):
    def __init__(self, min=0.3):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1, dilation=2)
        self.min = min

    def forward(self, x1):
        x1 = self.conv(x1)
        x1 = torch.clamp_min(x1, self.min)
        return x1

inputs_shape = [1, 32, 14, 14]
