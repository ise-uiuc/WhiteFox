
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 4, 2)
        self.conv2 = torch.nn.Conv1d(4, 2, 2)
        self.bn1 = torch.nn.BatchNorm1d(2)
        self.bn2 = torch.nn.BatchNorm1d(4)
        self.bn3 = torch.nn.BatchNorm1d(2)
    def forward(self, x6):
        x6 = self.conv(x6)
        x6 = self.bn1(x6)
        x6 = self.conv2(x6)
        x6 = self.bn2(x6)
        x6 = self.bn3(x6)
        y7 = x6
        return y7
# Inputs to the model
x6 = torch.randn(1, 2, 4)
