
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 12)
 
    def forward(self, x1):
        o = self.linear(x1)
        t1 = torch.relu(o)
        return t1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
