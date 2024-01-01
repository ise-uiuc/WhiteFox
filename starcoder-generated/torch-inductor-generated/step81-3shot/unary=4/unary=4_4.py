
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
 
    def forward(self, x1):
        y1 = self.linear(x1)
        y2 = y1 * 0.5
        y3 = y1 * 0.7071067811865476
        y4 = torch.erf(y3)
        y5 = y4 + 1
        y6 = y2 * y5
        return y6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
