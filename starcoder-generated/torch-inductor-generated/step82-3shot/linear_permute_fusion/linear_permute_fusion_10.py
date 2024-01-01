
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v2 = self.tanh.forward(v1)
        return v2 * x1 + x1 * v2
# Inputs to the model
x1 = torch.randn(1, 1, 2)
