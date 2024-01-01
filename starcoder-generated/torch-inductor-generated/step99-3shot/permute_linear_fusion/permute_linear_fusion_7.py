
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
    def forward(self, x1):
        v1 = x1.transpose(-1, -2).clone()
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        return v2
# Inputs to the model
x1 = torch.randn(1, 10, 10)
