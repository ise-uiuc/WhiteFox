
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(100, 200)
 
    def forward(self, x):
        v = self.linear1(x)
        v = v + x 
        return v

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(7, 100)
