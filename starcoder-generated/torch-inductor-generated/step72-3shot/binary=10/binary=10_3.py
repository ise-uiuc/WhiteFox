
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(100, 100, bias=False)
        self.linear2 = torch.nn.Linear(100, 100, bias=True)
 
    def forward(self, x1, x2):
        v1 = self.linear1(x1) # Transform input x1 with a linear transformation into a 100-dimensional vector
        v2 = self.linear2(x2) # Transform input x2 with a linear transformation into a 100-dimensional vector
        v3 = v1 + x2 # Add x2 to the output of the linear transformation applied to x1
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(100, 100)
x2 = torch.randn(100, 100)
