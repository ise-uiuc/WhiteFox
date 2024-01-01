
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        x2 = torch.matmul(v2, self.linear1.bias)
        z1 = x2 ** 3
        x3 = torch.matmul(z1, x2)
        v3 = torch.nn.functional.linear(v2, self.linear2.weight, self.linear2.bias)
        z2 = v1 / x3
        x3 = torch.nn.functional.linear(v3, self.linear1.weight, self.linear1.bias)
        x4 = x3 + z2
        z3 = x1 - z1
        return z1 + x2 / x3 + torch.matmul(x4, z3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
