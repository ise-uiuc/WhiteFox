
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x):
        v = self.linear(x)
        v = v + other
        return v

# Initializing the model
m = Model()

# Inputs to the model
