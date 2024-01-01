
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 1, (1, 0, 0), stride=(1, 0, 0), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 3.0
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1, 1)
