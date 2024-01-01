
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x):
        x = self.linear(x)
        return torch.sigmoid(x)

# Initializing the model
m = Model()

# Input to the model
x = torch.randn(2, 3)
y = m(x)

# Inputs to the model
x = torch.randn(2, 3)
