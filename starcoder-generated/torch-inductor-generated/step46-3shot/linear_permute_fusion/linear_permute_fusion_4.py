
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v2 = v1.permute(0, 2, 3, 1)
        v3 = v2.permute(0, 2, 1, 3)
        v4 = v2.permute(0, 3, 1, 2)
        v5 = torch.nn.functional.linear(v3, self.linear2.weight, self.linear2.bias)
        v6 = torch.nn.functional.linear(v4, self.linear2.weight, self.linear2.bias)
        return v5 * v6
# Inputs to the model
x1 = torch.randn(4, 2, 2, 2)
