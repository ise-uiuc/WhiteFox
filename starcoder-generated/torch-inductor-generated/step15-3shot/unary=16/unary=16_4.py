
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)
 
    def forward(self, x2):
        v1 = self.linear(x1)
        v2 = torch.relu(v1)
        return v2

# Initializing the model
m2 = Model()

# Inputs to the model
x2 = torch.randn(1, 8)
