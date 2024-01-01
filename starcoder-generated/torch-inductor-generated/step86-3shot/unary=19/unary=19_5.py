
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x):
        x = x.flatten(1)
        return torch.sigmoid(self.linear(x))

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 8)
