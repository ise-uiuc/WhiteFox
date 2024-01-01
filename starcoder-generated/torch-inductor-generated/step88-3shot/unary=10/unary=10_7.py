
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = x1.reshape(x1.size(0), -1)
        v2 = torch.nn.functional.linear(v1, weight=None, bias=3.0)
        v3 = torch.clamp(v2, 0.0, 6.0)
        v4 = v3/6
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 160, 7, 7)
