
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0 # Convert the results of the comparison into boolean values
        v3 = v1 * 0.01 # Use a small negative slope value to guarantee that some values become negative
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing a model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 10)
