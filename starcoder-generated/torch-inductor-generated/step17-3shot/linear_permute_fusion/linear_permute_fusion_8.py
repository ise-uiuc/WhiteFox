
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 6)
    def forward(self, x):
        v1 = x.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.permute(0, 2, 1)
        v4 = self.linear(v3)
        return v4
# Inputs to the model
x1 = torch.randn(2, 2, 2, device='cpu')
