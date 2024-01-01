
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 50, (1, 1), stride=(1, 1), padding=(0, 1))
    def forward(self, x):
        negative_slope = 0.097937
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.rand(3, 128, 22, 16)
