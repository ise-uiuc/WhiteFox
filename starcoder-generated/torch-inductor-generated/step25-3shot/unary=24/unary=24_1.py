
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act_fn = torch.nn.ELU
        self.conv = torch.nn.Conv2d(12, 13, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.act_fn()(self.conv(x))
        v2 = v1 > 0
        v3 = v1 * 0.1
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 12, 32, 32)
