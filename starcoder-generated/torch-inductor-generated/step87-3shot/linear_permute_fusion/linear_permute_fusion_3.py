
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x5, x1):
        v5 = x5.permute(1, 0, 2)
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v6 = v5.permute(1, 0, 2)
        return v6
# Inputs to the model
x5 = torch.randn(2, 1, 2)
x1 = torch.randn(1, 3, 2)
