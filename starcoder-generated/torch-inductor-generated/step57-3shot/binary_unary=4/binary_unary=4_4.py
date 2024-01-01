
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(10, 5)
        self.linear2 = torch.nn.Linear(5, 2)
 
    def forward(self, x, other):
        y = self.linear1(x)
        y = y + other
        return y

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(20, 10)
other = torch.randn(20, 5)
