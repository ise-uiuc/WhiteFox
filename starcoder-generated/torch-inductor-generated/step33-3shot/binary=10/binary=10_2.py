
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 8)
        self.fc2 = torch.nn.Linear(8, 8)
 
    def forward(self, x, y):
        z1 = self.fc1(x)
        z2 = z1 + x
        z3 = self.fc2(z1)
        z4 = z3 + x
        z5 = z2 * z4
        w1 = z2 - y
        w2 = w1 + z3
        w3 = w2 * z4
        return w3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.rand((10, 8))
y = torch.rand((10, 8))
