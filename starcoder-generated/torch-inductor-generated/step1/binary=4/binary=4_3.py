
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 4)
 
    def forward(self, x, other):
        x = self.linear1(x)
        x = x + other
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2)
other = torch.randn(1, 4)
