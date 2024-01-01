
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear3 = torch.nn.Linear(1, 1)
        self.linear2 = torch.nn.Linear(1, 2)
        self.linear = torch.nn.Linear(1, 2)
    def forward(self, x1, x2):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 1, 3, 2)
        v3 = torch.nn.functional.linear(v2, self.linear2.weight, self.linear2.bias)
        v4 = v3.permute(0, 1, 3, 2)
        v5 = torch.nn.functional.linear(x2, self.linear3.weight, self.linear3.bias)
        v6 = v5.permute(0, 1, 3, 2)
        v7 = v4.add(v6)
        return v4.sub(v7)
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)
x2 = torch.randn(1, 1, 2, 1)
