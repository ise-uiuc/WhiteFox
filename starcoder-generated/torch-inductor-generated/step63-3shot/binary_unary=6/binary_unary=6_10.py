
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)
 
    def forward(self, x):
        y = self.linear(x)
        y = y - 2
        y = torch.relu(y)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(4, 8)
