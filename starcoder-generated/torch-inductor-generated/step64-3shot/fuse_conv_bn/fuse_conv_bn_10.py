
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3, 3, 3)
        self.conv2 = torch.nn.Conv1d(3, 3, 3)
        self.bn = torch.nn.BatchNorm1d(3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.conv2(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 6)
