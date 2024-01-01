
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1):
        x2 = self.linear(x1)
        return x2

# Initializing the model
# Please initialize other with a scalar value.
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
