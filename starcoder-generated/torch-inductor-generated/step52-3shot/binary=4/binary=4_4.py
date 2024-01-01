
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6)
 
    def forward(self, x2):
        v2 = self.linear(x2)
        v3 = v2 + other
        return v3

# Initializing the model
m = Model()

# Inputs and additional input 'other'
x2 = torch.randn(1, 3)
other = torch.randn(1, 6)
