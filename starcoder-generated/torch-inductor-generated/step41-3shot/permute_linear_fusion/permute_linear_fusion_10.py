
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.cat((v1, v2), dim=-1)
        v4 = v3.permute(0, 2, -1, -2)
        v5 = v3 - v2
        return torch.matmul(v5, v4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
