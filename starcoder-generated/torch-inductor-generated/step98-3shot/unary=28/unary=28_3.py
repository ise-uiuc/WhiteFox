
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, min_value, max_value):
        v1 = torch.nn.functional.linear(x1, linear = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32), bias = torch.tensor([10, 20, 30], dtype=torch.float32))
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3
 
# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
min_value = torch.tensor(0.0, dtype=torch.float)
max_value = torch.tensor(1.0, dtype=torch.float)
