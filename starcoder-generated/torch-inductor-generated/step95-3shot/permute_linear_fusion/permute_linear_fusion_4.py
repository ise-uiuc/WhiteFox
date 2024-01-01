
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.tanh(x1)
        v2 = torch.autograd.grad(v1, x1, v1)
        v3 = torch.nn.functional.cos(self.linear.weight)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2, requires_grad=True)
