
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Conv1d(3, 4, 2, bias=False),
                                          torch.nn.BatchNorm1d(4, track_running_stats=False))
    def forward(self, x3):
        return self.layers(x3)
# Inputs to the model
x3 = torch.randn(1, 3, 4)
