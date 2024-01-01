
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x):
        v0 = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
        v1 = v0 + v0
        return v1
# Inputs to the model
x = torch.randn(1, 2, 2)
