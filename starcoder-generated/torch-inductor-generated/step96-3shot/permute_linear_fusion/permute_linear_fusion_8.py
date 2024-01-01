
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm1d(2)
        self.linear = torch.nn.Linear(2, 4)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.bn1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
