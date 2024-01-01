

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(784, 10)
 
    def forward(self, x, size):
        v1 = self.linear(x - size)
        return v1

# Initializing the model
m = Model()
size = 0.1

# Inputs to the model
x = torch.randn(1, 784)
