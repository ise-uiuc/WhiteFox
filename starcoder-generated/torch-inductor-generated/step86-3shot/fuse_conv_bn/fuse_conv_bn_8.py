
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(10)
        self.layer = torch.nn.Sequential(torch.nn.Conv1d(6, 2, 3, padding=1, stride=2, groups=4, bias=False), torch.nn.BatchNorm1d(2, affine=False, track_running_stats=False))
    def forward(self, x2):
        s2 = self.layer(x2)
        return s2 + s2
# Inputs to the model
x2 = torch.randn(1, 6, 5)
