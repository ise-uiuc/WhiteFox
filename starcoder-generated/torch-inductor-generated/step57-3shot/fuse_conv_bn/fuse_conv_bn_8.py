
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(19, 8, 2)
        self.bn = torch.nn.BatchNorm1d(8)
        torch.manual_seed(42)
        self.bn.weight = torch.nn.Parameter(torch.randn(14,))
        torch.manual_seed(56)
        self.bn.bias = torch.nn.Parameter(torch.randn(14,))
    def forward(self, x):
        x = self.bn(self.conv(x))
        return x
# Inputs to the model
x = torch.randn(1, 19, 15)
