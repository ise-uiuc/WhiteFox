
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
 
    def forward(self, x1):
        y1 = self.linear(x1)
        y2 = y1 + x1
        return y2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
other = torch.randn(1, 2)
