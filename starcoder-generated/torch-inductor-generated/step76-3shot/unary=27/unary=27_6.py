
class Model(torch.nn.Module):
    def __init__(self, min_weight, min_value):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.randn(3))
        self.conv = torch.nn.Conv2d(1, 3, 3, stride=2, padding=1)
        self.min_weight = min_weight
        self.min_value = min_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_weight)
        v3 = torch.nn.functional.conv2d(v2, torch.clamp_min(torch.clamp_max(self.conv.weight, self.min_weight), -self.min_weight), self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        v4 = torch.clamp_min(v3, self.min_value)
        v5 = torch.clamp_max(v4, -self.min_value)
        return v5
min_weight = 0.3
min_value = -0.8
# Inputs to the model
x1 = torch.randn(3, 1, 9, 10)
