
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(21, 10)
 
    def forward(self, x, other = 4):
        v1 = self.linear(x)
        v2 = v1 - other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 21)
