
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model and the "other" tensor supplied to the model's forward function.
x1 = torch.randn(2, 5)
other = torch.randn(2, 5)
