
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)
 
    def forward(self, x1, other):
        y = self.linear(x1)
        y2 = y + other
        y3 = torch.relu(y2)
        return y3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(64, 5)
other = torch.randn(64, 3)
