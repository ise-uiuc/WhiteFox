
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v3 = x1.shape
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 2, device='cpu')
