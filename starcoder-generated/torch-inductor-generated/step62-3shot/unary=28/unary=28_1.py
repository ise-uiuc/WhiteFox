
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, min_value=0.04603149747885704, max_value=0.21733654153823853):
        v1 = torch.nn.functional.linear(x1.__output__, self.weight, bias)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Initializing the model
m = Model()

# Inputs the model
x1 = torch.randn(1, 3, 64, 64)
