
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x):
        y = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
        out = y.permute(0, 3, 2, 1)
        return out
# Inputs to the model
x = torch.randn(2, 2, 2, 2)
