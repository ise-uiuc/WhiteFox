
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = torch.nn.functional.linear(input=x1, weight=self.weight, bias=self.bias)
        v2 = torch.clamp_min(v1, min_value=self.min_value)
        v3 = torch.clamp_max(v2, max_value=self.max_value)
        return v3

# Model parameters
Model.weight = torch.randn(8, 3, 64, 64)
Model.bias = torch.randn(8)
Model.min_value = -2.0
Model.max_value = 1.5

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
