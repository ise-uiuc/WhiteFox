
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1, x2):
        v4 = x1
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        v3 = x2
        v2 = v1.permute(0, 3, 1, 2)
        return (v1, v2, v3)
# Inputs to the model
x1 = torch.randn((1, 3, 3, 3), device='cpu')
x2 = torch.randn((1, 3, 3, 3), device='cpu')
