
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
 
    def forward(self, x1):
        y = self.linear(x1)
        z = y.relu()
        return z

# Initializing the model
model = Model()

# Input to the model
x1 = torch.randn(1, 2)
