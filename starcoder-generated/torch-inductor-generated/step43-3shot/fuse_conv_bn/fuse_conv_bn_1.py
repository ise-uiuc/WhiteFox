
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(6)
        self.layer = torch.nn.Sequential(torch.nn.Conv3d(3, 3, 3, bias=False), torch.nn.BatchNorm3d(3, track_running_stats=True))
    def forward(self, x):
        s = self.layer(x)
        return s
# Inputs to the model
x = torch.randn(1, 3, 8, 8, 8)
