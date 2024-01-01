
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(10, 10)
 
    def forward(self, x2):
        v2 = self.l(x2)
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(10, 10)
