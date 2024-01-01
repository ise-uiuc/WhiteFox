
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v1 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = torch.sum(v1)
        v3 = v1.permute(0, 2, 1)
        return v2 + v3
# Inputs to the model
x1 = torch.randn(1, 3, 3)
