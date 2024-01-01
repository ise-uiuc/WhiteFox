
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(3, 3)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = v2.permute(0, 2, 1)
        v4 = v3.permute(0, 2, 1)
        v4 = torch.nn.functional.linear(v4, self.linear2.weight, self.linear2.bias)
        x3 = torch.nn.functional.relu(v4)
        v5 = v4.permute(0, 2, 1)
        x2 = torch.matmul(x2, x3)
        v5 = x2.permute(0, 2, 1)
        v6 = x3.permute(0, 2, 1)
        v6 = torch.matmul(v5, v6)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
