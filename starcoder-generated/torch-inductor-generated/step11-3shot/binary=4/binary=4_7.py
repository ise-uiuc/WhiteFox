
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 16)
 
    def forward(self, x1, x2):
        y1 = torch.cat([x1, x2], 0)
        y2 = self.linear(y1)
        y3 = y2 + x2
        return y3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4)
x2 = torch.randn(2, 4)
