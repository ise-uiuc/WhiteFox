
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.linear_2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        x2 = torch.nn.functional.relu(x1)
        v1 = x2.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x3 = self.linear_2(x2)
        x4 = x3 * x2
        x5 = x3 + x2
        v3 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias) + x4
        x3 = x5.permute(0, 2, 1)
        v2 = torch.add(v2, v3)
        x4 = v2.permute(0, 2, 1)
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
