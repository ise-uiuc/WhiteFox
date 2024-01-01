
class Model(torch.nn.Module):
    bias = torch.nn.Parameter(torch.randn(8), requires_grad=True)
    def __init__(self, min_value=0.5, max_value=8, requires_grad=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
        self.bias = bias
    def forward(self, x1):
        v1 = self.bias + self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
