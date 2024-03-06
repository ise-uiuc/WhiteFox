
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v3 = v1.flatten(2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 2, 2)