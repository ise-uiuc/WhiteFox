
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.__linear__ = torch.nn.Linear(3, 8)
 
    def forward(self, input, x):
        y = self.__linear__(input)
        z = y + x
        return z

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(1, 3, 32, 32)
__other__ = torch.randn(1, 8, 32, 32)
