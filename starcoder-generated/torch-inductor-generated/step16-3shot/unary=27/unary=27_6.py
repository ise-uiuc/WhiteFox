
class Model(torch.nn.Module):
    def __init__(self, min_value=1.1):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 33, 1, stride=1, padding=0)
        self.min_value = min_value
    def forward(self, input):
        v1 = self.conv(input)
        v2 = torch.clamp_min(v1, self.min_value)
        return v2
# Inputs to the model
input = torch.randn(1, 5, 128, 128)
