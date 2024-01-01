
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3, track_running_stats=False)
    def forward(self, x1):
        v1 = self.bn(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
