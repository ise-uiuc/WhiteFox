
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)
 
    def forward(self, x2, other):
        v1 = self.linear(x2)
        
        # v1 should be added to other
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 5)
other = torch.randn(1, 3)
