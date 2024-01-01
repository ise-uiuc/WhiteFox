
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x1):
        v11 = x1.permute(0, 2, 1)
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        return v11 * v1
# Inputs to the model
x1 = torch.randn(2, 2, 2)
