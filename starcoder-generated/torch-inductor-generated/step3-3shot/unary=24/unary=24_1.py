
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.negative_slope = negative_slope
    def forward(self, x1):
        v1 = self.conv(x1)
        m1 = v1 > 0
        v2 = v1 * (-self.negative_slope)
        m2 = torch.tensor(0, dtype=bool)
        v3 = torch.where(m1, v2, m2)
        v4 = torch.sum(v3 / 0.5)
        return v4
negative_slope_range = [0.01, 0.05, 0.1, 0.5]
for negative_slope in negative_slope_range:
    mod = Model(negative_slope)
    mod_input = torch.randn(1, 3, 64, 64)
    mod(mod_input).sum().backward()
    print(mod.conv.weight.grad.max(), mod.conv.bias.grad.max())
