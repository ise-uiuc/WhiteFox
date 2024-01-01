
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
 
    def forward(self, x1):
        y = self.linear(x1)
        y = torch.sigmoid(y)
        return y

# Initializing the model
__model__ = Model()

# Inputs to the model
x1 = torch.randn(16, 3)
y = __model__(x1)

