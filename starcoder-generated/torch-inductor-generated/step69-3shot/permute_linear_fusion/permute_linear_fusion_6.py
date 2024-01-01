
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten(0, 1)
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.max(v2, dim=-1)[0]
        v4 = self.flatten(v3)
        x2 = torch.mul(self.linear.weight, self.linear.bias)
        v5 = torch.max(v4, x2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
