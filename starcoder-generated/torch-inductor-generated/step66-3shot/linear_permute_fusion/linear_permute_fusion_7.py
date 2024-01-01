
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
    def forward(self, x0):
        v0 = torch.nn.functional.linear(x0, self.linear.weight, self.linear.bias)
        v1 = x0 + 1.0
        v2 = v0 - v1.transpose(2, 1)
        return v2
# Inputs to the model
x0 = torch.randn(3, 3, 4)
