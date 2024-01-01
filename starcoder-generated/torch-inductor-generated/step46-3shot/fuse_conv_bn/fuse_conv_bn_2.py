
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(8, 8, 1)
        self.bn = torch.nn.BatchNorm1d(8)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(4)
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.bn(x)
        y = self.avg_pool(x)
        return y
# Inputs to the model
x = torch.randn(1, 8, 16)
