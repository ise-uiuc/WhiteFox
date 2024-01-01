,
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1.permute(0, 2, 1), self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        x2 = x1 + v2
        v3 = torch.nn.functional.linear(x2.permute(0, 2, 1), self.linear.weight, self.linear.bias)
        v4 = v3.permute(0, 2, 1)
        v5 = v4 + v2
        x3 = v5 + torch.nn.functional.relu(torch.matmul(v1, v3) + torch.nn.functional.relu(torch.matmul(v2, v4)))
        v6 = ((v5/2) ** 2).sum()
        v7 = torch.nn.functional.relu(torch.matmul(v3 + v2, v4) - v5 * v6).sum()
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 3)
