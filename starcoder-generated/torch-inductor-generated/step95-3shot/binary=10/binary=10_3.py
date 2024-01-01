
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 1, bias=True)
        self.linear2 = torch.nn.Linear(1, 1, bias=True)
 
    def forward(self, x1):
        v1 = self.linear1(x1) # Linear transformation
        v2 = v1 + self.linear2(x1) # Add the output of the first linear transformation and the output of the second linear transformation
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
