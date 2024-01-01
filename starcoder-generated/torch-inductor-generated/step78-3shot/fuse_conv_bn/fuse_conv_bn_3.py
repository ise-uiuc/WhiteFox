
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv1d(2, 2, 1) 
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm1d(2) 
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
# Inputs
x = torch.randn(1, 2, 2)
