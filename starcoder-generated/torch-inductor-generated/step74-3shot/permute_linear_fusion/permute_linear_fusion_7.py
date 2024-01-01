
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight * x2, self.linear.bias)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 3)
x2 = torch.randn(3, 1)
