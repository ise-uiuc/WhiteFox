
class Model(torch.nn.Module):
    def __init__(self, __min_value__, __max_value__):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, m.weight, m.bias)
        v2 = torch.clamp_max(v1, __max_value__)
        v3 = torch.clamp(__min_value__ + v2)
        return v3

max_value = -m.bias / (torch.sqrt(torch.sum(m.weight * m.weight)))
min_value = max_value * -1.2
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
