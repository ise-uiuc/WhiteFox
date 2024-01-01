
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 4, 3, groups=2)
        c = torch.nn.Conv1d(1, 2, 1)
        bn = torch.nn.BatchNorm1d(2)
        self.layer = torch.nn.Sequential(bn, c)
    def forward(self, x1):
        x1 = self.layer(x1)
        v1 = self.conv(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 4) 
