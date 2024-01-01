
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(torch.nn.ConvTranspose2d(1, 1, 1, bias=None), torch.nn.BatchNorm2d(1, affine=False, track_running_stats=True))
    def forward(self, x):
        x = self.layer(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 1, 1)
