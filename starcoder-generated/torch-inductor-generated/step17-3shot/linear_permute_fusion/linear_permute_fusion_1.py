
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    # Here `v3` is not used as an output to its own use, so it goes unused. Also `v4` is not used
    def forward(self, x1, x2, x3):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v4 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        return (v1, v2)
# Inputs to the model
x1 = torch.randn(2, 20, 2, device='cpu')
x2 = torch.randn(2, 20, 2, device='cpu')
x3 = torch.randn(2, 20, 2, device='cpu')
