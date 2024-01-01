
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1, x2):
        v1 = torch.add(x1, x2)
        v4 = x1
        v8 = self.linear.weight
        v3 = torch.nn.functional.linear(v4, v8, self.linear.bias)
        return torch.mul(v1, v3)
# Inputs to the model
x1 = torch.randn(1, 2, 2, device='cpu')
x2 = torch.randn(1, 2, 2, device='cpu')
