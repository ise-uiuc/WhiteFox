
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, negative_slope):
        t1 = x1.flatten(1)
        t2 = t1 > 0
        t3 = t1 * negative_slope
        t4 = torch.where(t2, t1, t3)
        return t4.view(x1.size())

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
negative_slope = 3.0
