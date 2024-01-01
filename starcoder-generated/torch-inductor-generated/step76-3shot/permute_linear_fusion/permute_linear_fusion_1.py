
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.linear1 = torch.nn.Linear(3, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v2 = x1 + 2.37
        v2 = torch.tanh(v2)
        v3 = 1.37 * x1 + 4.37 - 4.37
        v3 = torch.tanh(v3)
        return (v3, v1, v2)
# Inputs to the model
x1 = torch.randn(1, 3, 2)
