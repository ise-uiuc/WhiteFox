
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.linear_2 = torch.nn.Linear(3, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(1, 2, 0)
        result = self.linear_2.forward(v2)
        return result
# Inputs to the model
x1 = torch.randn(1, 2, 3, device='cpu')
