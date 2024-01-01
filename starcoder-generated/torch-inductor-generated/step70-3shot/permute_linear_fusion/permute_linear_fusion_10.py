
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.linear1 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.matmul(v2, x1)
        v3 = v3.permute(1, 0, 2)
        v4 = v3.sum(dim=-1)
        v5 = (v4 > 0).to(v4.dtype)
        v6 = v3.permute(0, 2, 1)
        v7 = torch.matmul(v5, v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 2)
