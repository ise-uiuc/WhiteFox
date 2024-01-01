
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(3, 50)
 
    def forward(self, x1):
        v1 = self.l1(x1)
        v2 = v1 > 0
        v3 = v1
        v4 = v1 * -1
        v5 = v4 + v2
        v6 = v3 * v5
        return v6

# Initializing the model
neg_slope = 0.5
__param__ = neg_slope   
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3)
