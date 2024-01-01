
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 16)
 
    def forward(self, x):
        x1 = self.linear(x)
        x2 = x1 + x
        out = torch.nn.functional.relu(x2)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 3)
