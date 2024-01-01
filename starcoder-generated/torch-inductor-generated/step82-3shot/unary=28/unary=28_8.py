
class Model1(torch.nn.Module):
    def __init__(self, *, min_value=-1.7320508075688772935274463415058723669, max_value=3.7320508075688772935274463415058723669):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, weight=torch.rand(12, 3), bias=torch.rand(12))
        v3 = torch.clamp(v1, min=min_value, max=max_value)
 
    return v3

# Initializing the model
m = Model1()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
