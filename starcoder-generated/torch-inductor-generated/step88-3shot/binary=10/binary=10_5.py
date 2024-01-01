
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = torch.nn.Linear(12, 12)
        self.lin1 = torch.nn.Linear(12, 12)
 
    def forward(self, x):
        v0 = self.lin0(x)
        v1 = self.lin1(v0)
        v2 = v1 + x
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 12)
