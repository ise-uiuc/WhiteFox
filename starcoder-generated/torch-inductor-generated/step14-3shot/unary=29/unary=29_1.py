
class Model(torch.nn.Module):
    def __init__(self, min_value=-4.2, max_value=-2.3, input_channels=3):
        super().__init__()
        t = torch.nn.ConvTranspose2d(input_channels, 8, 1, stride=1, padding=1)
        self.weight = torch.nn.Parameter(t.weight.data.transpose(1, 3))
        self.bias = torch.nn.Parameter(t.bias)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        t1 = torch.clamp_min(self.bias, self.min_value)
        t2 = torch.clamp_max(t1, self.max_value)
        v1 = torch.conv_transpose2d(x1, self.weight, bias=t2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
