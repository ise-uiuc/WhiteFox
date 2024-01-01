
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(2, 3, 2)
        self.conv2 = torch.nn.Conv1d(3, 2, 2)
        self.bn = torch.nn.BatchNorm1d(3)
    def forward(self, x):
        return self.bn(self.conv2(self.conv1(x)))
# Inputs to the model
x = torch.randn(1, 2, 3)
