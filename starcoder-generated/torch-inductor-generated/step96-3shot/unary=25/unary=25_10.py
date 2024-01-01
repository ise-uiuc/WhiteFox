
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.1):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
 
    def forward(self, x2):
        t1 = self.linear(x2)
        t2 = t1 > 0
        t3 = t1 * self.negative_slope
        t4 = torch.where(t2, t1, t3)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 8)
