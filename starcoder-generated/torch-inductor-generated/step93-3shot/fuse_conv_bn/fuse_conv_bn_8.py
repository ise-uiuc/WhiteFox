
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm1d(3)
    def forward(self, x1):
        a = torch.relu(torch.relu(input=x1))
        b = self.bn(a)
        return b
# Inputs to the model
x1 = torch.randn(1, 3, 6)
