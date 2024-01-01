
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
 
    def forward(self, x1, x2=None):
        x = self.linear(x1)
        x = x if x2 is None else x + x2
        h = torch.nn.functional.relu(x)
        return h

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.ones((1, 1))
x2 = torch.full((1, 2), 2)
