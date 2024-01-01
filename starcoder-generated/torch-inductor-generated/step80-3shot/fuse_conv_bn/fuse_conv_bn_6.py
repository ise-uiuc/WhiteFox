
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3, 3, 3)
        self.bn = torch.nn.BatchNorm1d(3, track_running_stats=True)
        self.conv2 = torch.nn.Conv1d(3, 3, 3)
        self.activation = torch.nn.ReLU()
    def forward(self, x1):
        s = self.conv1(x1)
        t = self.bn(s)
        y = self.conv2(t)
        return y
# Inputs to the model
x1 = torch.randn(1, 3, 6)
