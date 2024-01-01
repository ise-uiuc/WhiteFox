
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(23, 42)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 45
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Initializing the other var
other = torch.tensor(45.)

# Inputs to the model
x1 = torch.randn(1, 23)
