
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x2):
        v2 = self.linear(x2)
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m2 = Model()

# Inputs to the model
x2 = torch.randn(6, 3)
