
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(1, 2, 0)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.flatten()
        return torch.min(v3, dim=-1)[1]
# Inputs to the model
x1 = torch.randn(2, 2, 1)
