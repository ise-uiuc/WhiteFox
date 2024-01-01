
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.fc2 = torch.nn.Linear(16,9)
 
    def forward(self, x, other):
        x = self.fc1(x)
        y = self.fc2(x)
        y = y - other
        return y
x = torch.randn(4, 10)
other = torch.randn(4, 9)
Model()(x, other)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
other = torch.randn(1, 9)
