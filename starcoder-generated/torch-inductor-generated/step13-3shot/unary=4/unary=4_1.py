
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 128, bias=False)
 
    def forward(self, x1):
        y1 = self.linear(x1) # This is t1
        y2 = y1 * 0.5 # This is t2
        y3 = y1 * 0.7071067811865476 # This is t3
        y4 = torch.erf(y3) # This is t4
        y5 = y4 + 1 # This is t5
        y6 = y2 * y5 # This is t6
        return y6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
