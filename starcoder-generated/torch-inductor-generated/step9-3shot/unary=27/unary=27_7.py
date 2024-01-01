
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(2, 1, 2, stride=2, padding=2)
        self.min = min
        self.max = max
    def forward(self, input):
        v1 = self.conv_1(input)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = -0.5
max = 0.3
# Inputs to the model
input = torch.randn(1, 2, 50, 50)
