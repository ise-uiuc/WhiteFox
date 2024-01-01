
class Model(torch.nn.Module): # Using torch.nn.ReLU for Relu as an example
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
 
    def forward(self, x):
        return self.linear(x) - x

# Initializing the model

m = Model()

# Inputs to the model
x = torch.randn(2)

# This is the output of our `Model` model, it's supposed to be equivalent to the torch.nn.ReLU()
