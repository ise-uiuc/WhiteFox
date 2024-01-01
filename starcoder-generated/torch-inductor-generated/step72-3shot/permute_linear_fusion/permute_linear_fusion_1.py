
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.softplus1 = torch.nn.Softplus(beta=0.0875, threshold=16.0)
        self.softplus2 = torch.nn.Softplus(beta=18.0)
        self.hardsigmoid = torch.nn.Hardsigmoid()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.softplus1(v2)
        v4 = self.hardsigmoid(v3)
        return self.softplus2(v4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
