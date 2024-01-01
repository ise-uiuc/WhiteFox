
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.ConvTranspose1d(1, 1, 3)
        self.b = torch.nn.BatchNorm1d(1, affine=False, track_running_stats=False, momentum=0.0)
    def forward(self, x):
        o1 = self.a(x)
        o2 = self.b(o1)
        return o2
# Inputs to the model
x = torch.randn(1, 1, 3)
