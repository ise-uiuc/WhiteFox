
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.linear_1 = torch.nn.Linear(1, 1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = torch.nn.functional.linear(v1, self.linear_1.weight, self.linear_1.bias)
        v3 = self.tanh(v2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1)
