
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
 
    def forward(self, x):
        v = self.linear(x)
        r = v + x
        return r

# Initializing the model
m = Model()

# Inputs to the model
x = torch.tensor([[1.0, 0.0]])
