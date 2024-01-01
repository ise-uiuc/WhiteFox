
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 - x2
        v3 = v2.relu()
        return v3

# Initializing the model
m = Model()

# Initialization of the input tensor 'x1' and the second input tensor 'x2'
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 1)

# Inputs to the model
