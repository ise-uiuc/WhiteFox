
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        v2 = [2, 5, 2] # A hard-coded shape.
        v1 = torch.zeros([1] + v2)
        self.register_parameter('weight', torch.nn.Parameter(v1)) # Register a Parameter with the name "weight" and with the value as in v1.
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.weight, None)
        v2 = v1.permute(0, 2, 1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 3, 2, device='cpu')
