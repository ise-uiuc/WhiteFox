
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(24, 48)
 
    def forward(self, x):
        x = x + 1
        x = self.linear(x)
        x = torch.relu(x)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 24)
