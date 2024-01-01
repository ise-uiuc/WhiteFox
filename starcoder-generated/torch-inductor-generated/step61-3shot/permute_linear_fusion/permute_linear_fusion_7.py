
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 6)
        self.linear2 = torch.nn.Linear(2, 6)
        self.linear3 = torch.nn.Linear(2, 2)
        self.linear4 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        x2 = x1.permute(1, 2, 0)
        x3 = x1.permute(2, 1, 0)
        x4 = x3.permute(1, 2, 0)
        x5 = torch.nn.functional.linear(x4, self.linear1.weight, self.linear1.bias)
        x6 = torch.nn.functional.linear(x5, self.linear2.weight, self.linear2.bias)
        x7 = torch.nn.functional.linear(x6, self.linear3.weight, self.linear3.bias)
        return x7
# Inputs to the model
x1 = torch.randn(1, 2, 2)
