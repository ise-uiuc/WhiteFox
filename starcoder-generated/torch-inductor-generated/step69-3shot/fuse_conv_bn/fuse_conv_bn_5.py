
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(9)
        self.layer = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3), torch.nn.BatchNorm2d(3, track_running_stats=True))
    def forward(self, x):
        return self.layer(x)
# Inputs to the model
x = torch.randn(1, 3, 3, 3)
