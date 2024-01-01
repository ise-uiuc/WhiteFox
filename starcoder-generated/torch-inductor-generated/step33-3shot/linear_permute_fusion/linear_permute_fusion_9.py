
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v8 = self.linear.weight
        v2 = torch.nn.functional.linear(x1, v8, self.linear.bias)
        v3 = x1 - v2
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
