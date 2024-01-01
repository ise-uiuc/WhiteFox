
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, (1, 3), stride=(1, 2), padding=(0, 1))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add(1)
        v3 = v2.ceil()
        v4 = v3.div(1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
