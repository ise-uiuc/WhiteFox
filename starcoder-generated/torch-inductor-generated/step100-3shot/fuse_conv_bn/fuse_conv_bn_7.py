
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(4, 3, 1)
        torch.manual_seed(25)
        self.bn = torch.nn.BatchNorm1d(3, affine=True)
    def forward(self, x1):
        y1 = self.conv1(x1)
        y2 = self.bn(y1)
        return y2
# Inputs to the model
x1 = torch.randn(1, 4, 4)
