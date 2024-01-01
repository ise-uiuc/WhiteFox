
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.add = torch.nn.quantized.FloatFunctional()
        self.conv = torch.nn.Conv2d(1, 16, 7, stride=1, padding=3, groups=1, bias=True)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.relu(t1)
        t3 = torch.cat((t2, x1), axis=1)
        t4 = self.add.add_scalar(3, t3)
        t5 = torch.clamp_min(t4, 0)
        t6 = torch.clamp_max(t5, 6)
        t7 = t4 * t6
        t8 = t7 / 6.0
        return t8.unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
