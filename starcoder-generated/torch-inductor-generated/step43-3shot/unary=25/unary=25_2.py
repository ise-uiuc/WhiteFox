
negative_slope = 0.1

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 > 0
        t3 = t1 * negative_slope
        t4 = torch.where(t2, t1, t3)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
__input__ = torch.randn(1, 8)
