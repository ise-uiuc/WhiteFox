
class Model(torch.nn.Module):
    def __init__(self, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 2, stride=3, padding=1)
        self.max = max
    def forward(self, input):
        v1 = self.conv(input)
        v2 = torch.clamp_max(v1, self.max)
        return v2
# Inputs to the model
input = torch.randn(1, 3, 1024, 1024)

