
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = torch.nn.Linear(8, 16)
 
    def forward(self, x2):
        v2 = self.lin0(x2)
        v3 = torch.tanh(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 8)
