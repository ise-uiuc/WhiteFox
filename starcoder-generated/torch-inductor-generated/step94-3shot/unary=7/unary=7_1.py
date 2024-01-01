
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        l1 = torch.nn.functional.linear(x1, torch.ones((10,), dtype=torch.float32))
        l2 = l1 * torch.clamp(l1 + 3.0, min=0, max=6.0)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 10)
