
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 40)
 
    def forward(self, x):
        v = self.linear(x)
        v = torch.relu(v)
        return v

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
