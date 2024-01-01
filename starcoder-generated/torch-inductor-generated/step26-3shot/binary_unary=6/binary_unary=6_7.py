
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
        self.linear = torch.nn.Linear(2, 3)
 
    def forward(self, x):
        x = self.linear(x)
        y = torch.tensor([-17.44, 127.27, -1.53])
        x = x - y
        x = torch.nn.functional.relu(x)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2)
