
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 9)
        self.linear2 = torch.nn.Linear(8, 9)
 
    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(torch.relu(x))
        x = torch.relu(x)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
