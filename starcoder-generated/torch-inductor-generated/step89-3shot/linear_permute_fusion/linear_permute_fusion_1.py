
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        exp1 = torch.exp
        v2 = exp1(v1)
        v3 = v2.sum((-1, -2))
        v4 = v2.sum((-1, -2))
        v5 = v2.permute(1, 0, 2)
        return v3 + v4 + v5 * v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
