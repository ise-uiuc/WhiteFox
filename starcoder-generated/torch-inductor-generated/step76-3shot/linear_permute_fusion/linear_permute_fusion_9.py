
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(2, 3)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v0 = v1.mul(x1)
        v2 = torch.nn.functional.linear(x1, self.linear2.weight, self.linear2.bias).add(v0)
        v3 = v2.transpose(1, 2)
        return v3
# Inputs to the model
x2 = torch.randn(1, 2, 2)
