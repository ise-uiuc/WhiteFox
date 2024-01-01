
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x):
        return x + self.linear(x) + torch.zeros(10)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
