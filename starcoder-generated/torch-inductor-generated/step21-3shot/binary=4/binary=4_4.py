
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, input):
        return self.linear(input) + None

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(3, 10)
