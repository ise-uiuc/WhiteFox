
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 2)
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(1, 0, 2)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear1.bias)
        v3 = torch.matmul(v2, v2)
        v4 = torch.nn.functional.linear(v3, self.linear2.weight, self.linear2.bias)
        return v2
# Inputs to the model
x1 = torch.randn(2, 16, 1)
