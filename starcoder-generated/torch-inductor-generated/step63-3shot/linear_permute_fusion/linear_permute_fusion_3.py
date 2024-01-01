
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, x0, x1):
        v0 = torch.nn.functional.linear(x0, self.linear.weight, self.linear.bias)
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v0.permute(0, 2, 1) + v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v2, self.linear.weight[:-1] + self.linear.weight[-1:0:-1], self.linear.bias)
        v4 = torch.norm(v2 - v3, dim=1)
        return v4
# Inputs to the model
x0 = torch.randn(1, 2, 2)
x1 = torch.randn(1, 2, 2)
