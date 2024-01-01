
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 5)
    def forward(self, x):
        x = self.conv(x)
        y = torch.cat([x[..., 0::2], x[..., 1::2]], dim=-1)
        return y
# Inputs to the model
x = torch.randn(2, 3, 10, 10)
