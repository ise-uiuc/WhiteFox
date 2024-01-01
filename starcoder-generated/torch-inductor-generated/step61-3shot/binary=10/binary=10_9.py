
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Specify the linear transformation weights
        self.linear = torch.nn.Linear(3, 15)
 
    def forward(self, x1, other):
        v0 = self.linear(x1)
        v1 = v0 + other
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
