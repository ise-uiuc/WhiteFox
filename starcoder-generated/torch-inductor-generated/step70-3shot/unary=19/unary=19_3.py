
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(283, 10)
 
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 283)
