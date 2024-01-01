
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
    def forward(self, x1):
        v4 = torch.nn.functional.linear(x1-x1, self.linear.weight, self.linear.bias)
        v5 = False
        if v5:
            v1 = x1 - x1
            v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
            return v2
        else:
            v1 = x1 + x1
            v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
            return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
