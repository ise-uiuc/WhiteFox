
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2

# Initializing model
m = Model()

# Inputs: for tensor 1, we can use the following random tensor as input.
x1 = torch.randn(20, 5)
# For tensor 2, you can use the following random tensor as input.
x2 = torch.randn(10, 5)
# Outputs of the model
