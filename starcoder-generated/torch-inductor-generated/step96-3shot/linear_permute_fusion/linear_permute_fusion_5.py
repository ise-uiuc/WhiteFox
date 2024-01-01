
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v4 = x1
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 4, device='cpu', requires_grad=True)
