
class Model(torch.nn.Module):
    def __init__(self):
        super ().__init__()
        self.conv = torch.nn.Conv1d(3, 3, 1, stride=1, padding=1, groups=3)
        self.bn = torch.nn.BatchNorm1d(3)
        self.pool = torch.nn.GroupNorm(3, 3)
        self.pool2 = torch.nn.GroupNorm(3, 3)
    def forward(self, x1):
        