
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.sigmoid1 = torch.nn.Sigmoid()
    def forward(self, x1):
        x2 = x1.permute(0, 2, 1)
        x3 = x2.permute(2, 1, 0)
        x4 = x3.permute(1, 2, 0)
        x5 = torch.nn.functional.linear(x2, self.linear1.weight, self.linear1.bias)
        x6 = torch.nn.functional.linear(x5, self.linear2.weight, self.linear2.bias)
        x7 = self.sigmoid1(x6)
        return x7
# Inputs to the model
x1 = torch.randn(1, 2, 2)
