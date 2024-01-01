
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
 
    def forward(self, x):
        x1 = self.linear(x)
        x2 = torch.sigmoid(x1)
        return x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 2)
