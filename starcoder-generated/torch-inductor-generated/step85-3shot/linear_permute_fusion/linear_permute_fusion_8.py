
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
    def forward(self, x1, x2):
        v4 = x1
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 1)
x2 = torch.randn(2, 1)
