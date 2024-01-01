
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.max(v2, dim=-1)[0]
        v3 = x1.size(1)
        v3 = v3.float()
        return v2 / v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
