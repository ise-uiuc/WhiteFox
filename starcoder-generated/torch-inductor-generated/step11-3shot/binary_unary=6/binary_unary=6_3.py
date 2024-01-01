
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(64, 64, bias=False)
        self.relu = torch.nn.ReLU()
 
    def forward(self, x1):
        r1 = self.linear(x1)
        r2 = r1 - other
        r3 = self.relu(r2)
        return r3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 1)
