
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1, x2):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v2 = torch.nn.functional.linear(v1, self.linear2.weight, self.linear2.bias)
        v3 = v2.permute(0, 2, 1)
        v4 = torch.nn.functional.linear(x2, self.linear1.weight, self.linear1.bias)
        v5 = torch.nn.functional.linear(v4, self.linear2.weight, self.linear2.bias)
        v6 = v5.permute(0, 2, 1)
        return v3 + v6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
