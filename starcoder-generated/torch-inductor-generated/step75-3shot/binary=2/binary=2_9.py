
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 16, (3, 3), stride=(2, 2), dilation=(1, 1), padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 16.0
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
