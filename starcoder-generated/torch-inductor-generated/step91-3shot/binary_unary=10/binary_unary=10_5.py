
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 10)
 
    def forward(self, x):
        out = self.linear(x)
        out = out + x
        out = torch.relu(out)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x  = torch.randn(3)

# Output of the model
