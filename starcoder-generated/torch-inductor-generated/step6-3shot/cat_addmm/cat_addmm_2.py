
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm = torch.nn.Linear(64, 64)
 
    def forward(self, x1):
        x2 = self.mm(x1)
        return torch.cat([x1, x2], 1) # Add the result of the matrix multiplication

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = x1 * 2 # Multiply the input by 2
x3 = x2 + 1 # Add 1 to the multiplied input
