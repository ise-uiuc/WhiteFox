
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
    def forward(self, x):
        v1 = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
