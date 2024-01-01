
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3,1)

    def forward(self, x1):
        y = self.linear(x1)
        y = y - 755.0
        return y

# Initializing the model
m = Model() 

# Inputs to the model
x1 = torch.randn(1, 3)
