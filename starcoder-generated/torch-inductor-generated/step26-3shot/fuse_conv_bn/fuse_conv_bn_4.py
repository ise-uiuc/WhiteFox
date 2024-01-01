
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn18 = torch.nn.BatchNorm2d(3, track_running_stats=False)
    def forward(self, x1):
        v2 = self.conv(x1)
        v19 = v2
        for i in range(0, 5):
            v23 = self.bn18(v19)
            v29 = v23
            v37 = v23
        return v37
# Inputs to the model
x1 = torch.randn(1, 3, 1, 1)
