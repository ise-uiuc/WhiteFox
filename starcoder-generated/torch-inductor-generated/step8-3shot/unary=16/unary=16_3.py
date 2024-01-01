
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(15, 7)
 
    def forward(self, x2):
        a1 = self.lin(x2)
        a2 = F.relu(a1)
        return a2

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 15)
