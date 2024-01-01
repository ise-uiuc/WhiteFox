
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, None, bias=None)
        v2 = v1
        min_value = 0.5
        v3 = torch.clamp(v2, min_value)
        max_value = 0.7071067811865476
        v4 = torch.clamp(v3, max_value)
        return v4

# Initializing the model
m2 = Model2()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
