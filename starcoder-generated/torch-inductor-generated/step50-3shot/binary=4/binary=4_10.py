
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)
 
    def forward(self, x1, x2):
# The input x1 is passed to linear and the output is added to tensor x2
        v1 = self.linear(x1)
        return x2 + v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 100)
x2 = torch.randn(2, 100)
