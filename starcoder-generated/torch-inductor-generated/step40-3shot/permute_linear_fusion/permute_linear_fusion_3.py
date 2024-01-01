
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v5 = torch.randn_like(x1)
        v1 = x1.permute(-1, 0, -2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v4 = torch.mean(x2, dim=-1)
        v3 = v4.mean()
        return x2 + v3 + v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
