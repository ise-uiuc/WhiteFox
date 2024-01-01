
def custom_relu6(x):
    a = x + 3
    return torch.clamp(a, min=0, max=6) / 6
 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = F.linear(x1, torch.ones(4, 3))
        v2 = custom_relu6(v1)
        return v2

# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(1, 3)
