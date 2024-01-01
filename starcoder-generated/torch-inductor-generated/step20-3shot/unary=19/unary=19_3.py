
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lin = nn.Linear(100, 2)
 
    def forward(self, x1):
        o = self.lin(x1)
        o = torch.sigmoid(o)
        return o

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 100)
_output = m(x1)

