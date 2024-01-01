
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(1, 13, 3, stride=2, padding=1,  bias=True)
        self.min = min
        self.max = max
    def forward(self, input1):
        v1 = self.conv(input1)
        v2 = torch.clamp_min(v1, -5)
        v3 = torch.clamp_max(v2, 25)
        return v3
min = -30.26
max = 0.19
# Inputs to the model
x1 = torch.randn(2, 1, 192, 192)
