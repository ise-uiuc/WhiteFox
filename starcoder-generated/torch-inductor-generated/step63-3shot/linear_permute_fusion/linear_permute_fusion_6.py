
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1, False)
    def forward(self, x1):
        v0 = torch.nn.functional.linear(x1, self.linear.weight.permute(0, 2, 1), self.linear.bias)
        return v0
# Inputs to the model
x1 = torch.randn(1, 2, 2)
