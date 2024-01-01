
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.linear3 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v0 = self.linear1(x1)
        v1 = torch.nn.functional.linear(x1, self.linear2.weight, self.linear2.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = x1 - v2
        v4 = v0 + v3
        v5 = torch.nn.functional.linear(v4, self.linear3.weight, self.linear3.bias)
        v6 = x1 + v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 2)
