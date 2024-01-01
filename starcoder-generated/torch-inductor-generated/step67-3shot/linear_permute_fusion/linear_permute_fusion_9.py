
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5, device='cpu')
    def forward(self, x1, x2):
        v3 = x1
        v4 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        v5 = v4.permute(0, 2, 1)
        v6 = x2
        v7 = torch.nn.functional.linear(v6, self.linear.weight, self.linear.bias)
        return v5.permute(0, 2, 1) + v7.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 12, 5, device='cpu')
x2 = torch.randn(1, 12, 5, device='cpu')
