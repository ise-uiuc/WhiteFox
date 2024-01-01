
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.linear_2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.nn.functional.linear(v1, self.linear_2.weight, self.linear_2.bias)
        v4 = v2 * x1 + v3
        v5 = v4.permute(0, 2, 1)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
