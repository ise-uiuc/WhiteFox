
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Conv3d(1, 1, 1, bias=False, padding=2, dilation=2)
        self.b = torch.nn.BatchNorm3d(1, affine=True, track_running_stats=True)
    def forward(self, x):
        y = self.b(self.a(x))
        z = self.b(self.a(x))
        return (y, z)
# Inputs to the model
x = torch.randn(1, 1, 1, 5, 5)
