
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8)
 
def forward(self, x1, x2):
    v1 = self.linear(x1)
    v2 = v1 - x2
    v3 = torch.relu(v2)
    return v3

# Initializing the model
x2 = 0.5
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
