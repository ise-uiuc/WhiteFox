
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x0, x1):
        v0 = torch.nn.functional.linear(x0, self.linear.weight, self.linear.bias)
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v3 = torch.matmul(v0, v1.permute(0, 2, 1))
        return v3
# Inputs to the model
x0 = torch.randn(2, 2)
x1 = torch.randn(2, 2)
