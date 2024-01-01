
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
    def forward(self, x1):
        return torch.nn.functional.linear(x1.permute(0, 2, 1), self.linear.weight, self.linear.bias)
# Inputs to the model
x1 = torch.randn(1, 2, 4)
