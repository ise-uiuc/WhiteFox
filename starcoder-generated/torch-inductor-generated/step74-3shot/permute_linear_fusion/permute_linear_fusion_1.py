
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        a = torch.tanh(x1)
        v1 = a.permute(0, 2, 1)
        return torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
