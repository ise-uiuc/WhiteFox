
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
    def forward(self, x):
        v1 = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
        x2 = x + v1
        v3 = v1.permute(0, 2, 1)
        v4 = torch.nn.functional.linear(v3, self.linear.weight)
        return (x2 + v4) * x
# Inputs to the model
x = torch.randn(1, 3, 2)
