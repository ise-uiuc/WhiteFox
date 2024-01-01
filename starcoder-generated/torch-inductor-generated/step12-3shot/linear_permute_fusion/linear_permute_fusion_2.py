
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x1, x2):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v3 = x2
        v2 = v1.permute(1, 0, 2)
        v4 = x1
        v5 = v4.permute(1, 0, 2)
        return v1 + v2 + v3 + v5
# Inputs to the model
x1 = torch.randn(1, 1, 2)
x2 = torch.randn(2, 1, 2)
