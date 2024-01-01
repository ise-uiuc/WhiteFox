
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 50)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 - x2
        r3 = torch.nn.functional.relu(v2)
        return r3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100)
x2 = torch.randn(1, 100)
