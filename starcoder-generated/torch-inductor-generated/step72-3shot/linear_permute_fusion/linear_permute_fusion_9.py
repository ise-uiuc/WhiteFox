
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
        self.linear1 = torch.nn.Linear(3, 4)
        self.linear2 = torch.nn.Linear(4, 4)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v2, self.linear1.weight, self.linear1.bias)
        v4 = v3.permute(0, 2, 1)
        v5 = torch.nn.functional.linear(v4, self.linear2.weight, self.linear2.bias)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
