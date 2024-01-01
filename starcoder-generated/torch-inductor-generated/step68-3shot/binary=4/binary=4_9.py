
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, x2):
        y1 = self.linear(x1)
        y2 = y1 + x2
        y3 = y2 + x2
        y4 = y3 + x2
        y5 = y4 + x2
        y6 = y5 + x2
        y7 = y6 + x2
        y8 = y7 + x2
        return y8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 3)
