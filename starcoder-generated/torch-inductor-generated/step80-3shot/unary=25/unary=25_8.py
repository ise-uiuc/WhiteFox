
class Model(torch.nn.Module):
    def __init__(self, in_ch, out_ch, negative_slope=0.2):
        super().__init__()
        self.linear = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model(3, 8)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
