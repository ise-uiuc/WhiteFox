
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 > 0
        t3 = t1 * self.negative_slope
        t4 = torch.where(t2, t1, t3)
        return t4

# Initializing the model
m = Model(0.5)

# Inputs to the model
x1 = torch.randn(1, 8)
