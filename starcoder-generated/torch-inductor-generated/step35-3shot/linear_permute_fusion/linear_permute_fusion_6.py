
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1, x2):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        v3 = v1.permute(0, 2, 1)
        v4 = v2.permute(0, 2, 1)
        v6 = v4 * v3.max() + v3
        return torch.sigmoid(v6)
# Inputs to the model
x1 = torch.randn(2, 2, 2)
x2 = torch.randn(2, 2, 2)
