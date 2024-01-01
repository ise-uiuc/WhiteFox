
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1):
        out = F.linear(x1, 4, 3, 1)
        out = F.hardsigmoid(out, offset=3, slope=0.2)
        out = F.hardsigmoid(out, offset=0, slope=1)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
